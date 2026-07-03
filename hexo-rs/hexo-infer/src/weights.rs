//! Safetensors weight loading + model-config validation for hexo-infer.
//!
//! The safetensors file is produced by `hexo_a0.export.save_safetensors` and carries
//! `__metadata__`: `format` ("hexo-safetensors-v1"), `model_config` (JSON), `train_steps`,
//! `source_checkpoint`, and optionally `game_config`. Tensor names are the `HeXONet`
//! state_dict keys with `_orig_mod.` stripped.

#![allow(dead_code)]

const FORMAT_TAG: &str = "hexo-safetensors-v1";

#[derive(Debug)]
pub enum InferError {
    BadFormat(String),
    UnsupportedConfig(String),
    MissingTensor(String),
    BadShape(String),
    BadDtype(String),
    Search(&'static str),
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferError::BadFormat(m) => write!(f, "bad safetensors format: {m}"),
            InferError::UnsupportedConfig(m) => write!(f, "unsupported model config: {m}"),
            InferError::MissingTensor(m) => write!(f, "missing tensor: {m}"),
            InferError::BadShape(m) => write!(f, "bad tensor shape: {m}"),
            InferError::BadDtype(m) => write!(f, "bad dtype: {m}"),
            InferError::Search(m) => write!(f, "search error: {m}"),
        }
    }
}
impl std::error::Error for InferError {}
impl From<&'static str> for InferError {
    fn from(s: &'static str) -> Self {
        InferError::Search(s)
    }
}

/// Resolved model config (parsed from safetensors metadata).
#[derive(Debug, Clone)]
pub struct InferConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub node_dim: usize,
    pub use_jk_cat: bool,
    pub policy_hidden: usize,
    pub value_hidden: usize,
    pub prune_empty_edges: bool,
    pub threat_features: bool,
    pub relative_stones: bool,
    pub source_checkpoint: String,
    pub metadata_json: String,
}

/// One GINE layer's weights.
#[derive(Debug, Clone)]
pub struct GineLayer {
    pub eps: f32,
    pub lin_w: Vec<f32>, pub lin_b: Vec<f32>,       // (H,H), (H)
    pub nn0_w: Vec<f32>, pub nn0_b: Vec<f32>,       // (H,H), (H)
    pub nn2_w: Vec<f32>, pub nn2_b: Vec<f32>,       // (H,H), (H)
    pub norm_w: Vec<f32>, pub norm_b: Vec<f32>,     // (H), (H)
}

/// All loaded weights + resolved config.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub config: InferConfig,
    pub input_proj_w: Vec<f32>, pub input_proj_b: Vec<f32>,   // (H, node_dim), (H)
    pub edge_proj_w: Vec<f32>, pub edge_proj_b: Vec<f32>,     // (H, 5), (H)
    pub layers: Vec<GineLayer>,
    pub final_norm_w: Vec<f32>, pub final_norm_b: Vec<f32>,   // (H), (H)
    pub policy0_w: Vec<f32>, pub policy0_b: Vec<f32>,         // (P, D), (P)  D = L*H (cat) or H
    pub policy2_w: Vec<f32>, pub policy2_b: Vec<f32>,         // (1, P), (1)
    pub value0_w: Vec<f32>, pub value0_b: Vec<f32>,           // (V, D), (V)
    pub value2_w: Vec<f32>, pub value2_b: Vec<f32>,           // (1, V), (1)
}

impl ModelWeights {
    pub fn source_checkpoint(&self) -> &str {
        &self.config.source_checkpoint
    }

    pub fn from_safetensors(bytes: &[u8]) -> Result<ModelWeights, InferError> {
        use safetensors::SafeTensors;
        // read_metadata parses just the header (cheap) for the __metadata__ map;
        // deserialize gives the SafeTensors for tensor access.
        let (_n, hdr) = SafeTensors::read_metadata(bytes)
            .map_err(|e| InferError::BadFormat(e.to_string()))?;
        let custom = hdr.metadata().as_ref().ok_or_else(|| {
            InferError::BadFormat("no __metadata__ (not a hexo-safetensors-v1 file)".into())
        })?;
        let fmt = custom.get("format").ok_or_else(|| {
            InferError::BadFormat("missing 'format' metadata key".into())
        })?;
        if fmt != FORMAT_TAG {
            return Err(InferError::BadFormat(format!("expected {FORMAT_TAG}, got {fmt}")));
        }
        let mc_json = custom.get("model_config").ok_or_else(|| {
            InferError::BadFormat("missing 'model_config' metadata".into())
        })?;
        let config = parse_config(mc_json, custom)?;
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| InferError::BadFormat(e.to_string()))?;

        let h = config.hidden_dim;
        let l = config.num_layers;
        let d = if config.use_jk_cat { l * h } else { h }; // head input dim
        let nd = config.node_dim;
        let (ph, vh) = (config.policy_hidden, config.value_hidden);

        let mut layers = Vec::with_capacity(l);
        for i in 0..l {
            layers.push(GineLayer {
                eps: tensor_f32(&st, &format!("representation.convs.{i}.eps"), &[1])?[0],
                lin_w: tensor_f32(&st, &format!("representation.convs.{i}.lin.weight"), &[h, h])?,
                lin_b: tensor_f32(&st, &format!("representation.convs.{i}.lin.bias"), &[h])?,
                nn0_w: tensor_f32(&st, &format!("representation.convs.{i}.nn.0.weight"), &[h, h])?,
                nn0_b: tensor_f32(&st, &format!("representation.convs.{i}.nn.0.bias"), &[h])?,
                nn2_w: tensor_f32(&st, &format!("representation.convs.{i}.nn.2.weight"), &[h, h])?,
                nn2_b: tensor_f32(&st, &format!("representation.convs.{i}.nn.2.bias"), &[h])?,
                norm_w: tensor_f32(&st, &format!("representation.norms.{i}.weight"), &[h])?,
                norm_b: tensor_f32(&st, &format!("representation.norms.{i}.bias"), &[h])?,
            });
        }

        Ok(ModelWeights {
            config,
            input_proj_w: tensor_f32(&st, "representation.input_proj.weight", &[h, nd])?,
            input_proj_b: tensor_f32(&st, "representation.input_proj.bias", &[h])?,
            edge_proj_w: tensor_f32(&st, "representation.edge_proj.weight", &[h, 5])?,
            edge_proj_b: tensor_f32(&st, "representation.edge_proj.bias", &[h])?,
            layers,
            final_norm_w: tensor_f32(&st, "representation.final_norm.weight", &[h])?,
            final_norm_b: tensor_f32(&st, "representation.final_norm.bias", &[h])?,
            policy0_w: tensor_f32(&st, "policy_head.mlp.0.weight", &[ph, d])?,
            policy0_b: tensor_f32(&st, "policy_head.mlp.0.bias", &[ph])?,
            policy2_w: tensor_f32(&st, "policy_head.mlp.2.weight", &[1, ph])?,
            policy2_b: tensor_f32(&st, "policy_head.mlp.2.bias", &[1])?,
            value0_w: tensor_f32(&st, "value_head.mlp.0.weight", &[vh, d])?,
            value0_b: tensor_f32(&st, "value_head.mlp.0.bias", &[vh])?,
            value2_w: tensor_f32(&st, "value_head.mlp.2.weight", &[1, vh])?,
            value2_b: tensor_f32(&st, "value_head.mlp.2.bias", &[1])?,
        })
    }
}

/// Parse the model_config JSON + validate against hexo-infer's supported subset.
fn parse_config(
    mc_json: &str,
    custom: &std::collections::HashMap<String, String>,
) -> Result<InferConfig, InferError> {
    let v: serde_json::Value =
        serde_json::from_str(mc_json).map_err(|e| InferError::BadFormat(format!("model_config JSON: {e}")))?;
    let get = |k: &str, d: &serde_json::Value| -> serde_json::Value {
        v.get(k).cloned().unwrap_or_else(|| d.clone())
    };
    let bool_of = |k: &str, default: bool| -> bool {
        get(k, &serde_json::Value::Bool(default)).as_bool().unwrap_or(default)
    };
    let str_of = |k: &str, default: &str| -> String {
        get(k, &serde_json::Value::String(default.into()))
            .as_str()
            .unwrap_or(default)
            .to_string()
    };
    let usize_of = |k: &str, default: usize| -> usize {
        get(k, &serde_json::Value::Number(default.into()))
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(default)
    };

    let conv_type = str_of("conv_type", "gine");
    let graph_type = str_of("graph_type", "axis");
    let pre_norm = bool_of("pre_norm", true);
    let use_layer_scale = bool_of("use_layer_scale", false);
    let use_jk = bool_of("use_jk", false);
    let jk_mode = str_of("jk_mode", "sum");
    let threat = bool_of("threat_features", false);
    let relative = bool_of("relative_stone_encoding", false);
    let prune = bool_of("prune_empty_edges", true);

    // Loud rejection of unsupported configs (the parity suite only covers what ships).
    if conv_type != "gine" {
        return Err(InferError::UnsupportedConfig(format!("conv_type={conv_type:?} (only 'gine' supported)")));
    }
    if graph_type != "axis" {
        return Err(InferError::UnsupportedConfig(format!("graph_type={graph_type:?} (only 'axis' supported)")));
    }
    if !pre_norm {
        return Err(InferError::UnsupportedConfig("pre_norm=false (only pre_norm=true supported)".into()));
    }
    if use_layer_scale {
        return Err(InferError::UnsupportedConfig("use_layer_scale=true (not supported)".into()));
    }
    // JK: cat (use_jk=true) OR no-JK (use_jk=false). sum/max/lstm rejected.
    if use_jk && jk_mode != "cat" {
        return Err(InferError::UnsupportedConfig(format!("jk_mode={jk_mode:?} (only 'cat' supported when use_jk)")));
    }

    let hidden_dim = usize_of("hidden_dim", 256);
    let num_layers = usize_of("num_layers", 3);
    let policy_hidden = usize_of("policy_hidden", 128);
    let value_hidden = usize_of("value_hidden", 128);
    let node_dim = (if relative { 7 } else { 8 }) + (if threat { 4 } else { 0 });
    let use_jk_cat = use_jk && jk_mode == "cat";

    Ok(InferConfig {
        hidden_dim,
        num_layers,
        node_dim,
        use_jk_cat,
        policy_hidden,
        value_hidden,
        prune_empty_edges: prune,
        threat_features: threat,
        relative_stones: relative,
        source_checkpoint: custom.get("source_checkpoint").cloned().unwrap_or_default(),
        metadata_json: serde_json::to_string(custom).unwrap_or_default(),
    })
}

/// Read a tensor as f32, validating dtype + shape (row-major, PyTorch Linear (out,in)).
fn tensor_f32(
    st: &safetensors::SafeTensors,
    name: &str,
    expect: &[usize],
) -> Result<Vec<f32>, InferError> {
    use safetensors::Dtype;
    let tv = st
        .tensor(name)
        .map_err(|e| InferError::MissingTensor(format!("{name}: {e}")))?;
    if tv.dtype() != Dtype::F32 {
        return Err(InferError::BadDtype(format!("{name}: expected F32, got {:?}", tv.dtype())));
    }
    let shape = tv.shape();
    if shape != expect {
        return Err(InferError::BadShape(format!(
            "{name}: expected {expect:?}, got {shape:?}"
        )));
    }
    let bytes = tv.data();
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    #[test]
    fn tiny_model_loads() {
        let bytes = std::fs::read(fixtures_dir().join("tiny.safetensors")).unwrap();
        let w = ModelWeights::from_safetensors(&bytes).unwrap();
        assert_eq!(w.config.hidden_dim, 16);
        assert_eq!(w.config.num_layers, 2);
        assert_eq!(w.config.node_dim, 11); // relative(7) + threat(4)
        assert!(w.config.use_jk_cat);
        assert_eq!(w.config.policy_hidden, 16);
        assert_eq!(w.config.value_hidden, 8);
        assert_eq!(w.input_proj_w.len(), 16 * 11);
        assert_eq!(w.policy0_w.len(), 16 * (2 * 16)); // P*D, D=L*H=32
        assert_eq!(w.source_checkpoint(), "tiny");
    }

    #[test]
    fn tiny_nojk_model_loads() {
        let bytes = std::fs::read(fixtures_dir().join("tiny_nojk.safetensors")).unwrap();
        let w = ModelWeights::from_safetensors(&bytes).unwrap();
        assert!(!w.config.use_jk_cat);
        assert_eq!(w.config.node_dim, 8); // absolute, no threat
        assert_eq!(w.policy0_w.len(), 16 * 16); // D=H (no jk cat)
    }

    fn doctored_safetensors(metadata_json: &str) -> Vec<u8> {
        // Build a minimal valid safetensors with one zero tensor + doctored __metadata__.
        // Header: {"name":{dtype:F32,shape:[1],data_offsets:[0,4]},"__metadata__":{...}}
        let header = serde_json::json!({
            "zero": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "__metadata__": serde_json::from_str::<serde_json::Value>(metadata_json).unwrap(),
        })
        .to_string();
        let header_bytes = header.as_bytes();
        let mut out = Vec::with_capacity(8 + header_bytes.len() + 4);
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&[0u8; 4]);
        out
    }

    #[test]
    fn rejects_unsupported_configs() {
        for (bad, mc) in [
            (r#"{"conv_type":"gatv2"}"#, "conv_type"),
            (r#"{"graph_type":"hex"}"#, "graph_type"),
            (r#"{"pre_norm":false}"#, "pre_norm"),
            (r#"{"use_layer_scale":true}"#, "layer_scale"),
            (r#"{"use_jk":true,"jk_mode":"sum"}"#, "jk_mode"),
            (r#"{"use_jk":true,"jk_mode":"max"}"#, "jk_mode"),
            (r#"{"use_jk":true,"jk_mode":"lstm"}"#, "jk_mode"),
        ] {
            // Valid model_config with the bad field overlaid INTO model_config.
            let mut mc_val: serde_json::Value = serde_json::json!({
                "hidden_dim":16,"num_layers":2,"conv_type":"gine","graph_type":"axis",
                "pre_norm":true,"use_jk":true,"jk_mode":"cat","policy_hidden":16,
                "value_hidden":8,"threat_features":true,"relative_stone_encoding":true,
                "prune_empty_edges":true,
            });
            let overlay: serde_json::Value = serde_json::from_str(bad).unwrap();
            if let (serde_json::Value::Object(m), serde_json::Value::Object(o)) = (&mut mc_val, overlay) {
                for (k, v) in o { m.insert(k.clone(), v); }
            }
            // model_config is a JSON string inside __metadata__.
            let meta = serde_json::json!({
                "format": "hexo-safetensors-v1",
                "model_config": mc_val.to_string(),
                "train_steps": "0", "source_checkpoint": "test",
            });
            let bytes = doctored_safetensors(&meta.to_string());
            let err = ModelWeights::from_safetensors(&bytes).unwrap_err();
            assert!(matches!(err, InferError::UnsupportedConfig(_)),
                "{bad}: expected UnsupportedConfig, got {err}");
            assert!(err.to_string().contains(mc), "{bad}: err {err} missing {mc}");
        }
    }

    #[test]
    fn rejects_bad_format() {
        let bytes = doctored_safetensors(r#"{"format":"something-else","model_config":"{}","train_steps":"0","source_checkpoint":"x"}"#);
        assert!(matches!(
            ModelWeights::from_safetensors(&bytes).unwrap_err(),
            InferError::BadFormat(_)
        ));
    }
}
// WebWorker wrapper around StrixBot. Message protocol:
//   in : {type:"init", weightsUrl}                                -> {type:"ready", info}
//   in : {type:"best_move", position, sims, mActions, seed}       -> {type:"move", result, ms}
//   in : {type:"evaluate", position}                              -> {type:"eval", result}
// Any failure -> {type:"error", error}
import init, { StrixBot } from "../pkg/hexo_wasm.js";

let bot = null;
self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      await init();
      const weights = new Uint8Array(await (await fetch(msg.weightsUrl)).arrayBuffer());
      bot = new StrixBot(weights);
      self.postMessage({ type: "ready", info: JSON.parse(bot.model_info()) });
    } else if (msg.type === "best_move") {
      const t0 = performance.now();
      const result = JSON.parse(bot.best_move(JSON.stringify(msg.position), msg.sims, msg.mActions, BigInt(msg.seed ?? 0)));
      self.postMessage({ type: "move", result, ms: performance.now() - t0 });
    } else if (msg.type === "evaluate") {
      const result = JSON.parse(bot.evaluate(JSON.stringify(msg.position)));
      self.postMessage({ type: "eval", result });
    }
  } catch (e) {
    self.postMessage({ type: "error", error: String(e) });
  }
};
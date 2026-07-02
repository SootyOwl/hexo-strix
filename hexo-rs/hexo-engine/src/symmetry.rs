use crate::types::Coord;

/// The 12 symmetry transforms of the D6 group (hexagonal dihedral symmetry).
/// Each function maps an axial coordinate (q, r) to its transformed counterpart.
pub const D6_TRANSFORMS: [fn(Coord) -> Coord; 12] = [
    // Rotations
    |(q, r)| (q, r),           // 0: identity
    |(q, r)| (-r, q + r),     // 1: rot60
    |(q, r)| (-q - r, q),     // 2: rot120
    |(q, r)| (-q, -r),        // 3: rot180
    |(q, r)| (r, -q - r),     // 4: rot240
    |(q, r)| (q + r, -q),     // 5: rot300
    // Reflections
    |(q, r)| (r, q),          // 6: reflect
    |(q, r)| (-q, q + r),     // 7: ref+60
    |(q, r)| (-q - r, r),     // 8: ref+120
    |(q, r)| (-r, -q),        // 9: ref+180
    |(q, r)| (q, -q - r),     // 10: ref+240
    |(q, r)| (q + r, -r),     // 11: ref+300
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex::hex_distance;
    use std::collections::HashSet;

    #[test]
    fn all_12_transforms_are_distinct() {
        let input = (2, 1);
        let results: HashSet<Coord> = D6_TRANSFORMS.iter().map(|t| t(input)).collect();
        assert_eq!(results.len(), 12);
    }

    #[test]
    fn identity_is_identity() {
        for (q, r) in [(0, 0), (1, 2), (-3, 5), (7, -4)] {
            assert_eq!(D6_TRANSFORMS[0]((q, r)), (q, r));
        }
    }

    #[test]
    fn rot60_six_times_returns_to_origin() {
        let rot60 = D6_TRANSFORMS[1];
        let start = (2, 1);
        let mut c = start;
        for _ in 0..6 {
            c = rot60(c);
        }
        assert_eq!(c, start);
    }

    #[test]
    fn reflect_is_involution() {
        let reflect = D6_TRANSFORMS[6];
        for coord in [(0, 0), (2, 1), (-3, 5), (7, -4)] {
            assert_eq!(reflect(reflect(coord)), coord);
        }
    }

    #[test]
    fn origin_maps_to_origin() {
        let origin = (0, 0);
        for t in &D6_TRANSFORMS {
            assert_eq!(t(origin), origin);
        }
    }

    #[test]
    fn transforms_preserve_hex_distance() {
        let pairs = [((2, 1), (0, 0)), ((3, -1), (1, 2)), ((-2, 4), (5, -3))];
        for (a, b) in pairs {
            let expected = hex_distance(a, b);
            for t in &D6_TRANSFORMS {
                assert_eq!(
                    hex_distance(t(a), t(b)),
                    expected,
                    "transform changed distance between {a:?} and {b:?}"
                );
            }
        }
    }
}

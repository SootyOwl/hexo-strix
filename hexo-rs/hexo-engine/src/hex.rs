use crate::types::Coord;

/// Returns all (dq, dr) offsets within hex-distance ≤ `radius` of the origin.
/// Used for legal move generation and cache updates.
pub fn hex_offsets(radius: i32) -> Vec<Coord> {
    let mut offsets = Vec::new();
    for dq in -radius..=radius {
        for dr in -radius..=radius {
            if dq.abs().max(dr.abs()).max((dq + dr).abs()) <= radius {
                offsets.push((dq, dr));
            }
        }
    }
    offsets
}

/// Returns the hex distance between two axial coordinates.
/// Calculated as `max(|dq|, |dr|, |dq + dr|)`.
pub fn hex_distance(a: Coord, b: Coord) -> i32 {
    let dq = (b.0 - a.0).abs();
    let dr = (b.1 - a.1).abs();
    let ds = (b.0 - a.0 + b.1 - a.1).abs();
    dq.max(dr).max(ds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_point_is_zero() {
        assert_eq!(hex_distance((0, 0), (0, 0)), 0);
        assert_eq!(hex_distance((3, -2), (3, -2)), 0);
    }

    #[test]
    fn all_six_adjacents_are_distance_one() {
        let origin = (0, 0);
        let neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];
        for n in neighbors {
            assert_eq!(hex_distance(origin, n), 1, "neighbor {n:?} should be distance 1");
        }
    }

    #[test]
    fn along_axis_distance() {
        // Moving 5 steps along the q-axis
        assert_eq!(hex_distance((0, 0), (5, 0)), 5);
        // Moving 4 steps along the r-axis
        assert_eq!(hex_distance((0, 0), (0, 4)), 4);
        // Moving 3 steps along the s-axis (q + r = const)
        assert_eq!(hex_distance((0, 0), (3, -3)), 3);
    }

    #[test]
    fn diagonal_distance() {
        // (2, 1): dq=2, dr=1, dq+dr=3 → max = 3
        assert_eq!(hex_distance((0, 0), (2, 1)), 3);
        // (1, 2): dq=1, dr=2, dq+dr=3 → max = 3
        assert_eq!(hex_distance((0, 0), (1, 2)), 3);
    }

    #[test]
    fn distance_is_symmetric() {
        assert_eq!(hex_distance((1, 2), (4, -1)), hex_distance((4, -1), (1, 2)));
        assert_eq!(hex_distance((-3, 5), (2, -2)), hex_distance((2, -2), (-3, 5)));
    }

    #[test]
    fn negative_coords() {
        // (-2, -3) to (1, 1): dq=3, dr=4, dq+dr=7 → max = 7
        assert_eq!(hex_distance((-2, -3), (1, 1)), 7);
    }

    #[test]
    fn boundary_distance_eight() {
        // (0,0) to (8,0): along q-axis, distance 8
        assert_eq!(hex_distance((0, 0), (8, 0)), 8);
    }

    #[test]
    fn boundary_distance_nine() {
        // (0,0) to (9,0): along q-axis, distance 9
        assert_eq!(hex_distance((0, 0), (9, 0)), 9);
    }
}

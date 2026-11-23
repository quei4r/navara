//! Pole of inaccessibility — where a polygon's label goes.

use std::collections::BinaryHeap;

/// The point inside a polygon farthest from its boundary, to within
/// `precision`, as MapLibre places a polygon's symbol (Mapbox's `polylabel`).
///
/// `rings[0]` is the outer ring and the rest are holes, in any planar units;
/// a ring may or may not repeat its first vertex. Unlike a centroid, the result
/// is always inside the polygon, and away from narrow parts of it. A polygon
/// with no area answers its bounding box's minimum corner; one with no vertices
/// answers `None`.
pub fn pole_of_inaccessibility<R: AsRef<[(f64, f64)]>>(
    rings: &[R],
    precision: f64,
) -> Option<(f64, f64)> {
    let outer = rings.first()?.as_ref();
    let (first, rest) = outer.split_first()?;
    let (mut min, mut max) = (*first, *first);
    for &(x, y) in rest {
        min = (min.0.min(x), min.1.min(y));
        max = (max.0.max(x), max.1.max(y));
    }
    // A polygon narrower than `precision` would seed `long side / short side`
    // cells, and any point inside it is within `precision` of the pole.
    let cell_size = (max.0 - min.0).min(max.1 - min.1);
    if cell_size <= precision {
        return Some(interior_point(rings).unwrap_or(min));
    }

    let cell = |x: f64, y: f64, half: f64| Cell::new(x, y, half, rings);
    let half = cell_size / 2.0;
    let mut queue = BinaryHeap::new();
    let mut x = min.0;
    while x < max.0 {
        let mut y = min.1;
        while y < max.1 {
            queue.push(cell(x + half, y + half, half));
            y += cell_size;
        }
        x += cell_size;
    }

    let mut best = centroid_cell(outer, rings);
    let center = cell((min.0 + max.0) / 2.0, (min.1 + max.1) / 2.0, 0.0);
    if center.distance > best.distance {
        best = center;
    }

    // The queue pops the cell that could hold the farthest point first, so
    // once that one cannot beat `best` by more than `precision`, none can.
    while let Some(c) = queue.pop() {
        if c.distance > best.distance {
            best = c;
        }
        if c.potential - best.distance <= precision {
            break;
        }
        let half = c.half / 2.0;
        for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            queue.push(cell(c.x + dx * half, c.y + dy * half, half));
        }
    }
    // Only within `precision` of the pole: where no inscribed circle is wider
    // than that, the search can settle outside the polygon.
    if best.distance > 0.0 {
        return Some((best.x, best.y));
    }
    Some(interior_point(rings).unwrap_or(min))
}

/// A point strictly inside the polygon (even-odd over every ring, as
/// [`signed_distance`] tests it), or `None` when it has no area.
///
/// The middle of the widest span a horizontal line cuts out of the polygon,
/// with the line midway between two neighbouring vertex heights: it then
/// passes through no vertex and along no edge, so every crossing is a clean
/// one and the span between a crossing pair is inside.
fn interior_point<R: AsRef<[(f64, f64)]>>(rings: &[R]) -> Option<(f64, f64)> {
    let mut heights: Vec<f64> = rings
        .iter()
        .flat_map(|r| r.as_ref().iter().map(|p| p.1))
        .collect();
    heights.sort_by(f64::total_cmp);
    heights.dedup();
    let y = heights
        .windows(2)
        .max_by(|a, b| (a[1] - a[0]).total_cmp(&(b[1] - b[0])))
        .map(|w| (w[0] + w[1]) * 0.5)?;

    let mut crossings = Vec::new();
    for ring in rings {
        let ring = ring.as_ref();
        let Some(&last) = ring.last() else { continue };
        let mut b = last;
        for &a in ring {
            if (a.1 > y) != (b.1 > y) {
                crossings.push((b.0 - a.0) * (y - a.1) / (b.1 - a.1) + a.0);
            }
            b = a;
        }
    }
    crossings.sort_by(f64::total_cmp);
    crossings
        .as_chunks::<2>()
        .0
        .iter()
        .max_by(|a, b| (a[1] - a[0]).total_cmp(&(b[1] - b[0])))
        .filter(|span| span[1] > span[0])
        .map(|span| ((span[0] + span[1]) * 0.5, y))
}

/// A square of the search grid, ordered by the farthest a point inside it
/// could possibly be from the boundary.
#[derive(Clone, Copy)]
struct Cell {
    x: f64,
    y: f64,
    half: f64,
    /// Signed distance from the center to the boundary, positive inside.
    distance: f64,
    /// Upper bound on `distance` anywhere in the cell.
    potential: f64,
}

impl Cell {
    fn new<R: AsRef<[(f64, f64)]>>(x: f64, y: f64, half: f64, rings: &[R]) -> Self {
        let distance = signed_distance((x, y), rings);
        Self {
            x,
            y,
            half,
            distance,
            potential: distance + half * std::f64::consts::SQRT_2,
        }
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.potential.total_cmp(&other.potential).is_eq()
    }
}

impl Eq for Cell {}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.potential.total_cmp(&other.potential)
    }
}

/// The outer ring's area centroid as a starting guess, or its first vertex
/// when the ring has no area.
fn centroid_cell<R: AsRef<[(f64, f64)]>>(outer: &[(f64, f64)], rings: &[R]) -> Cell {
    let mut area = 0.0;
    let (mut cx, mut cy) = (0.0, 0.0);
    let mut prev = outer[outer.len() - 1];
    for &p in outer {
        let f = p.0 * prev.1 - prev.0 * p.1;
        cx += (p.0 + prev.0) * f;
        cy += (p.1 + prev.1) * f;
        area += f * 3.0;
        prev = p;
    }
    if area == 0.0 {
        return Cell::new(outer[0].0, outer[0].1, 0.0, rings);
    }
    Cell::new(cx / area, cy / area, 0.0, rings)
}

/// Distance from `p` to the nearest ring edge, positive inside the polygon
/// (even-odd over every ring, so holes count as outside).
fn signed_distance<R: AsRef<[(f64, f64)]>>(p: (f64, f64), rings: &[R]) -> f64 {
    let mut inside = false;
    let mut min_sq = f64::INFINITY;
    for ring in rings {
        let ring = ring.as_ref();
        let Some(&last) = ring.last() else { continue };
        let mut b = last;
        for &a in ring {
            if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
                inside = !inside;
            }
            min_sq = min_sq.min(segment_distance_sq(p, a, b));
            b = a;
        }
    }
    let d = min_sq.sqrt();
    if inside { d } else { -d }
}

/// Squared distance from `p` to the segment `a`–`b`.
fn segment_distance_sq(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (mut x, mut y) = a;
    let (dx, dy) = (b.0 - x, b.1 - y);
    if dx != 0.0 || dy != 0.0 {
        let t = ((p.0 - x) * dx + (p.1 - y) * dy) / (dx * dx + dy * dy);
        if t > 1.0 {
            (x, y) = b;
        } else if t > 0.0 {
            x += dx * t;
            y += dy * t;
        }
    }
    (p.0 - x).powi(2) + (p.1 - y).powi(2)
}

#[cfg(test)]
mod test {
    use super::*;

    fn square(min: f64, max: f64) -> Vec<(f64, f64)> {
        vec![(min, min), (max, min), (max, max), (min, max), (min, min)]
    }

    #[test]
    fn square_labels_at_its_center() {
        let (x, y) = pole_of_inaccessibility(&[square(0.0, 10.0)], 0.01).unwrap();
        assert!((x - 5.0).abs() < 0.1 && (y - 5.0).abs() < 0.1, "({x}, {y})");
    }

    #[test]
    fn open_and_closed_rings_agree() {
        let closed = square(0.0, 10.0);
        assert_eq!(
            pole_of_inaccessibility(&[&closed[..]], 0.01),
            pole_of_inaccessibility(&[&closed[..4]], 0.01),
        );
    }

    #[test]
    fn hole_pushes_the_label_off_center() {
        let rings = [square(0.0, 10.0), square(3.0, 7.0)];
        let p = pole_of_inaccessibility(&rings, 0.01).unwrap();
        // Inside the ring band, and not hugging either of its edges.
        assert!(signed_distance(p, &rings) > 1.0, "{p:?}");
    }

    #[test]
    fn concave_polygon_labels_inside_its_widest_part() {
        // An L whose area centroid falls outside the polygon, in the notch.
        let l = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 2.0),
            (2.0, 2.0),
            (2.0, 10.0),
            (0.0, 10.0),
        ];
        let p = pole_of_inaccessibility(&[&l], 0.01).unwrap();
        assert!(signed_distance(p, &[&l]) > 0.9, "{p:?}");
    }

    #[test]
    fn polygon_without_area_answers_its_minimum_corner() {
        let line = vec![(1.0, 5.0), (4.0, 5.0), (1.0, 5.0)];
        assert_eq!(pole_of_inaccessibility(&[line], 0.01), Some((1.0, 5.0)));
    }

    #[test]
    fn slivers_narrower_than_precision_label_inside() {
        // Each one's bounding-box corner is outside it, or on its edge.
        let cases: [Vec<Vec<(f64, f64)>>; 4] = [
            // Narrower than precision across its bounding box.
            vec![vec![(0.0, 0.5), (1000.0, 0.0), (1000.0, 0.5)]],
            vec![vec![(0.0, 0.0), (1000.0, 0.0), (1000.0, 1e-7), (0.0, 1e-7)]],
            // A wide bounding box around a thin diagonal: the search runs.
            vec![vec![(0.0, 0.0), (1000.0, 999.5), (1000.0, 1000.0)]],
            // A thin frame, whose centre is in its hole.
            vec![square(0.0, 100.0), square(0.25, 99.75)],
        ];
        for rings in cases {
            let p = pole_of_inaccessibility(&rings, 1.0).unwrap();
            assert!(signed_distance(p, &rings) > 0.0, "{p:?} outside {rings:?}");
        }
    }

    #[test]
    fn polygon_without_vertices_answers_none() {
        let empty: [Vec<(f64, f64)>; 1] = [Vec::new()];
        assert_eq!(pole_of_inaccessibility(&empty, 0.01), None);
        let none: [Vec<(f64, f64)>; 0] = [];
        assert_eq!(pole_of_inaccessibility(&none, 0.01), None);
    }
}

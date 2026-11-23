//! Cluster-continuation flags emitted by the shaper, which the TS layout pass
//! relies on to put letter spacing between clusters only.

use navara_wasm_font_worker::shaping::{CHAR_CLASS_NEWLINE, shape_text};

const MONO_FONT: &[u8] = include_bytes!("fixtures/demo_monochrome.ttf");

#[test]
fn separate_characters_start_their_own_clusters() {
    let glyphs = shape_text(MONO_FONT, "Hello").expect("shape");
    assert_eq!(glyphs.len(), 5);
    assert!(glyphs.iter().all(|g| !g.continues_cluster));
}

#[test]
fn combining_marks_continue_their_base_cluster() {
    // Two stacked marks: no font has a precomposed form for this, so the
    // shaper must emit the base plus mark glyphs in one grapheme cluster.
    let glyphs = shape_text(MONO_FONT, "a\u{0301}\u{0302}b").expect("shape");
    assert!(
        glyphs.len() >= 3,
        "expected base + marks + b, got {glyphs:?}"
    );
    assert!(!glyphs[0].continues_cluster);
    let last = glyphs.len() - 1;
    assert!(glyphs[1..last].iter().all(|g| g.continues_cluster));
    assert!(!glyphs[last].continues_cluster, "`b` starts a new cluster");
}

#[test]
fn newline_resets_cluster_tracking() {
    let glyphs = shape_text(MONO_FONT, "a\nb").expect("shape");
    assert_eq!(glyphs.len(), 3);
    assert_eq!(glyphs[1].char_class, CHAR_CLASS_NEWLINE);
    assert!(glyphs.iter().all(|g| !g.continues_cluster));
}

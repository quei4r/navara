/// Character classes attached to each shaped glyph so the TS layout pass can
/// find line-break opportunities without mapping glyphs back to source text.
/// Values cross the WASM boundary as raw `u8` — keep in sync with
/// `GlyphCharClass` in `web/navara_font/src/types.ts`.
pub const CHAR_CLASS_NORMAL: u8 = 0;
/// Breakable whitespace: a line may wrap here, dropping the glyph.
pub const CHAR_CLASS_WHITESPACE: u8 = 1;
/// Synthetic hard-break marker emitted between `\n`-separated segments.
/// Carries no glyph (zero advance); never enters the atlas.
pub const CHAR_CLASS_NEWLINE: u8 = 2;
/// Scripts written without word-separating spaces (CJK ideographs, kana,
/// hangul, fullwidth forms): a line may break after this glyph.
pub const CHAR_CLASS_IDEOGRAPHIC: u8 = 3;

/// Direction of the `\n`-separated segment a glyph was shaped in. Values cross
/// the WASM boundary as raw `u8` — keep in sync with `GlyphDirection` in
/// `web/navara_font/src/types.ts`.
///
/// No character with a script of its own (digits, punctuation, spaces only):
/// shaped left-to-right, but says nothing about the paragraph's direction.
/// Font-family fallback shapes such runs apart from the letters around them,
/// so layout takes a line's direction from its first non-neutral glyph.
pub const DIRECTION_NEUTRAL: u8 = 0;
pub const DIRECTION_LTR: u8 = 1;
/// Shaped right-to-left: the segment's glyphs arrive in visual order
/// (reversed logical order).
pub const DIRECTION_RTL: u8 = 2;

/// A single shaped glyph with positioning info from harfbuzz.
#[derive(Debug, Clone)]
pub struct ShapedGlyph {
    /// Glyph ID in the font (also used to look up atlas rect)
    pub glyph_id: u32,
    /// Horizontal advance after this glyph (in font units)
    pub x_advance: i32,
    /// Vertical advance after this glyph (in font units)
    pub y_advance: i32,
    /// Horizontal offset before drawing (in font units)
    pub x_offset: i32,
    /// Vertical offset before drawing (in font units)
    pub y_offset: i32,
    /// One of the `CHAR_CLASS_*` constants, derived from the source character
    /// that produced this glyph (the cluster's first char for ligatures).
    pub char_class: u8,
    /// True when this glyph belongs to the same cluster as the glyph before
    /// it (combining marks, pre-base vowels, multi-glyph decompositions).
    /// Layout inserts letter spacing only between clusters, so a mark never
    /// drifts off its base.
    pub continues_cluster: bool,
    /// One of the `DIRECTION_*` constants, shared by every glyph of the
    /// `\n`-separated segment.
    pub direction: u8,
}

/// Vertical line metrics in font units (from the hhea table).
#[derive(Debug, Clone, Copy)]
pub struct LineMetrics {
    pub ascender: i16,
    pub descender: i16,
    pub line_gap: i16,
}

/// Get the units-per-em value from raw font data.
pub fn get_units_per_em(font_data: &[u8]) -> Option<u16> {
    let face = rustybuzz::Face::from_slice(font_data, 0)?;
    Some(face.units_per_em() as u16)
}

/// Get vertical line metrics from raw font data.
pub fn get_line_metrics(font_data: &[u8]) -> Option<LineMetrics> {
    let face = rustybuzz::Face::from_slice(font_data, 0)?;
    Some(LineMetrics {
        ascender: face.ascender(),
        descender: face.descender(),
        line_gap: face.line_gap(),
    })
}

fn classify_char(c: char) -> u8 {
    if c.is_whitespace() {
        return CHAR_CLASS_WHITESPACE;
    }
    if is_ideographic(c) {
        return CHAR_CLASS_IDEOGRAPHIC;
    }
    CHAR_CLASS_NORMAL
}

/// Characters after which a line may break without whitespace, following the
/// convention used by other map renderers for CJK text.
fn is_ideographic(c: char) -> bool {
    matches!(c as u32,
        0x2E80..=0x303F   // CJK radicals, Kangxi radicals, CJK symbols/punct
        | 0x3040..=0x30FF // hiragana, katakana
        | 0x3130..=0x318F // hangul compatibility jamo
        | 0x31C0..=0x31EF // CJK strokes
        | 0x3200..=0x9FFF // enclosed CJK, CJK ext A, unified ideographs
        | 0xAC00..=0xD7AF // hangul syllables
        | 0xF900..=0xFAFF // CJK compatibility ideographs
        | 0xFE30..=0xFE4F // CJK compatibility forms
        | 0xFF00..=0xFF60 // fullwidth forms
        | 0x20000..=0x2FA1F // CJK ext B+ and supplement
    )
}

/// Whether letter spacing may be applied to `text`. Tracking pulls apart the
/// joined letterforms of cursive Arabic, so (like MapLibre) any Arabic
/// character disables it for the whole label.
pub fn allows_letter_spacing(text: &str) -> bool {
    !text.chars().any(is_cursive_arabic)
}

/// Every Arabic block: Arabic, Supplement, Extended-A/B/C and the presentation
/// forms. MapLibre exempts all but Extended-B/C (newer blocks it predates),
/// which hold joining letters too.
fn is_cursive_arabic(c: char) -> bool {
    matches!(c as u32,
        0x0600..=0x06FF   // Arabic
        | 0x0750..=0x077F // Arabic Supplement
        | 0x0870..=0x08FF // Arabic Extended-B, Extended-A
        | 0xFB50..=0xFDFF // Arabic Presentation Forms-A
        | 0xFE70..=0xFEFF // Arabic Presentation Forms-B
        | 0x10EC0..=0x10EFF // Arabic Extended-C
    )
}

/// Shape a text string using rustybuzz (harfbuzz port).
///
/// Each `\n`-separated segment is shaped in its own buffer so the newline
/// control character never reaches the shaper (no tofu) and shaping state
/// cannot leak across a forced break. A zero-advance marker glyph with
/// `CHAR_CLASS_NEWLINE` is emitted between segments so the layout pass still
/// sees the break position in the flat glyph stream.
///
/// Returns positioned glyph info.
/// Use the `units_per_em` cached on `FontEntry` for unit conversion.
///
pub fn shape_text(font_data: &[u8], text: &str) -> Option<Vec<ShapedGlyph>> {
    let face = rustybuzz::Face::from_slice(font_data, 0)?;

    let mut glyphs = Vec::new();
    for (i, segment) in text.split('\n').enumerate() {
        if i > 0 {
            glyphs.push(ShapedGlyph {
                glyph_id: 0,
                x_advance: 0,
                y_advance: 0,
                x_offset: 0,
                y_offset: 0,
                char_class: CHAR_CLASS_NEWLINE,
                continues_cluster: false,
                direction: DIRECTION_NEUTRAL,
            });
        }
        if segment.is_empty() {
            continue;
        }

        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(segment);
        buffer.guess_segment_properties();
        let direction = if buffer.script() == rustybuzz::script::UNKNOWN {
            DIRECTION_NEUTRAL
        } else if buffer.direction() == rustybuzz::Direction::RightToLeft {
            DIRECTION_RTL
        } else {
            DIRECTION_LTR
        };

        let output = rustybuzz::shape(&face, &[], buffer);

        let infos = output.glyph_infos();
        let positions = output.glyph_positions();

        let mut prev_cluster = None;
        glyphs.extend(infos.iter().zip(positions.iter()).map(|(info, pos)| {
            // Clusters are byte offsets into `segment` (valid for RTL runs
            // too, where clusters arrive in descending order).
            let char_class = segment[info.cluster as usize..]
                .chars()
                .next()
                .map(classify_char)
                .unwrap_or(CHAR_CLASS_NORMAL);
            let continues_cluster = prev_cluster == Some(info.cluster);
            prev_cluster = Some(info.cluster);
            ShapedGlyph {
                glyph_id: info.glyph_id,
                x_advance: pos.x_advance,
                y_advance: pos.y_advance,
                x_offset: pos.x_offset,
                y_offset: pos.y_offset,
                char_class,
                continues_cluster,
                direction,
            }
        }));
    }

    Some(glyphs)
}

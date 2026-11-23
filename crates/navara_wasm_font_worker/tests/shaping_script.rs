//! Script-derived facts the shaper reports so the TS layout pass never
//! inspects source text: direction per `\n` segment, and whether letter
//! spacing is allowed.

use navara_wasm_font_worker::shaping::{
    CHAR_CLASS_NEWLINE, DIRECTION_LTR, DIRECTION_NEUTRAL, DIRECTION_RTL, allows_letter_spacing,
    shape_text,
};

const MONO_FONT: &[u8] = include_bytes!("fixtures/demo_monochrome.ttf");

fn direction(text: &str) -> u8 {
    let glyphs = shape_text(MONO_FONT, text).expect("shape");
    assert!(!glyphs.is_empty());
    let direction = glyphs[0].direction;
    assert!(
        glyphs.iter().all(|g| g.direction == direction),
        "{text:?} mixes directions"
    );
    direction
}

#[test]
fn detects_rtl_scripts() {
    assert_eq!(direction("شارع الملك"), DIRECTION_RTL);
    assert_eq!(direction("רחוב"), DIRECTION_RTL);
}

#[test]
fn latin_and_cjk_are_ltr() {
    assert_eq!(direction("Main St"), DIRECTION_LTR);
    assert_eq!(direction("東京都"), DIRECTION_LTR);
}

#[test]
fn first_script_character_decides_direction() {
    assert_eq!(direction("Cafe شارع"), DIRECTION_LTR);
    assert_eq!(direction("شارع Cafe"), DIRECTION_RTL);
    // Digits and punctuation carry no script.
    assert_eq!(direction("12 - شارع"), DIRECTION_RTL);
}

#[test]
fn text_without_a_script_is_neutral() {
    assert_eq!(direction("123 !?"), DIRECTION_NEUTRAL);
}

#[test]
fn each_line_has_its_own_direction() {
    let glyphs = shape_text(MONO_FONT, "Tokyo\nشارع").expect("shape");
    let newline = glyphs
        .iter()
        .position(|g| g.char_class == CHAR_CLASS_NEWLINE)
        .expect("newline marker");
    assert!(
        glyphs[..newline]
            .iter()
            .all(|g| g.direction == DIRECTION_LTR)
    );
    assert!(
        glyphs[newline + 1..]
            .iter()
            .all(|g| g.direction == DIRECTION_RTL)
    );
}

#[test]
fn letter_spacing_allowed_for_non_arabic_scripts() {
    assert!(allows_letter_spacing("Main St"));
    assert!(allows_letter_spacing("東京都"));
    assert!(allows_letter_spacing("רחוב"));
}

#[test]
fn letter_spacing_disallowed_for_any_arabic() {
    assert!(!allows_letter_spacing("شارع"));
    assert!(!allows_letter_spacing("Cafe شارع"));
    // Joining letters from Extended-B and Extended-C, the two blocks
    // MapLibre's list predates.
    assert!(!allows_letter_spacing("\u{0870}"));
    assert!(!allows_letter_spacing("\u{10EC2}"));
}

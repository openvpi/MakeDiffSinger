# Identify reusable source fields

Identify each field by its role, including when a file combines several fields or two fields use the same writing system.

| Field | Meaning |
| --- | --- |
| Written lyrics | The text intended to be sung, as written in the main lyric line or a verified lyric field. Kana or Latin letters can themselves be lyrics. |
| Reading notation | A proposed or intended reading such as furigana, pinyin, or a romanized reading. Kana above a word can be reading notation. |
| Pronunciation | The way a passage is spoken or sung. A source annotation can propose a reading, G2P can generate alternatives, and audio can support a realization. Record the evidence for each. |
| Phoneme sequence | Ordered sound units in the selected inventory. Written `花` can have the reading `はな` and the phones `h a n a`. |
| Grouping | The mapping from phones to units such as pronunciation groups from forced alignment or syllables for note extraction. These units can have different boundaries. |
| Timing | Boundaries or durations attached to lyrics, phones, groups, syllables, or notes. A label and its timing are separately reusable. |
| Notes | Note boundaries, pitch, and rest or presence labels. |

Record the adopted fields and the fields to regenerate. A request to reuse only lyrics allows the main written text. Extract ruby base text and the written text from PFML. If the document has been flattened, verify which text is the main lyric line. Check the actual lyric matcher or forced aligner input, including higher-priority sidecars, for excluded material.

Keep excluded fields out of inputs, candidate generation, pronunciation repair, and validation. Generate requested fields from the allowed inputs. If the fields cannot be separated reliably, report the affected input and continue with usable material.

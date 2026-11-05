//! Store-private indices for full, 128-bit business entity identities.
//!
//! Both the allocator and command recorder use randomized aHash keys. Their
//! maps remain independent: one tracks live locations, the other merges a
//! batch's commands. Neither hash values nor iteration order are persisted;
//! equality, conflict checks and generational validation remain unchanged.

use crate::EntityId;

pub(crate) type EntityIndex<V> =
    std::collections::HashMap<EntityId, V, ahash::RandomState>;

const _: () = assert!(
    std::mem::size_of::<EntityId>() == 16 && std::mem::align_of::<EntityId>() == 1
);

pub(crate) mod entity_slice_serde {
    use crate::EntityId;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    const DELTA_RLE_MAGIC: u8 = 0xFD;
    const TAG_RUN_U64: u8 = 0x01;
    const TAG_RUN_U128: u8 = 0x02;
    const TAG_DELTAS_U64: u8 = 0x03;
    const TAG_DELTAS_U128: u8 = 0x04;

    #[inline(always)]
    fn delta_at(slice: &[EntityId], idx: usize) -> u128 {
        slice[idx].to_u128().wrapping_sub(slice[idx - 1].to_u128())
    }

    #[inline]
    fn run_len_from(slice: &[EntityId], start: usize, d0: u128) -> usize {
        let mut len = 1usize;
        let mut prev = slice[start].to_u128();
        for item in &slice[start + 1..] {
            let cur = item.to_u128();
            if cur.wrapping_sub(prev) != d0 {
                break;
            }
            len += 1;
            prev = cur;
        }
        len
    }

    pub(crate) fn encode_constant_stride_run(
        len: usize,
        first: u128,
        stride: u128,
    ) -> Option<Vec<u8>> {
        if len < 4 {
            return None;
        }
        let total_count = u32::try_from(len).ok()?;
        let run_len = (len - 1) as u32;
        if stride <= u64::MAX as u128 {
            let mut buf = Vec::with_capacity(34);
            buf.push(DELTA_RLE_MAGIC);
            buf.extend_from_slice(&total_count.to_le_bytes());
            buf.extend_from_slice(&first.to_le_bytes());
            buf.push(TAG_RUN_U64);
            buf.extend_from_slice(&run_len.to_le_bytes());
            buf.extend_from_slice(&(stride as u64).to_le_bytes());
            Some(buf)
        } else {
            let mut buf = Vec::with_capacity(42);
            buf.push(DELTA_RLE_MAGIC);
            buf.extend_from_slice(&total_count.to_le_bytes());
            buf.extend_from_slice(&first.to_le_bytes());
            buf.push(TAG_RUN_U128);
            buf.extend_from_slice(&run_len.to_le_bytes());
            buf.extend_from_slice(&stride.to_le_bytes());
            Some(buf)
        }
    }

    pub(crate) fn encode_delta_rle(slice: &[EntityId]) -> Option<Vec<u8>> {
        if slice.len() < 4 {
            return None;
        }
        let Ok(total_count) = u32::try_from(slice.len()) else {
            return None;
        };
        let first = slice[0].to_u128();
        let second = slice[1].to_u128();
        let stride = second.wrapping_sub(first);

        // Fast-path check for a single constant-stride run across the entire column.
        let is_single_run = slice[2..]
            .iter()
            .enumerate()
            .all(|(i, id)| id.to_u128() == first.wrapping_add(stride.wrapping_mul((i + 2) as u128)));
        if is_single_run {
            return encode_constant_stride_run(slice.len(), first, stride);
        }

        let raw_byte_len = slice.len().checked_mul(16)?;
        let mut buf = Vec::with_capacity(256.min(raw_byte_len));
        buf.push(DELTA_RLE_MAGIC);
        buf.extend_from_slice(&total_count.to_le_bytes());
        buf.extend_from_slice(&first.to_le_bytes());

        let mut idx = 1usize;
        while idx < slice.len() {
            let d0 = delta_at(slice, idx);
            let run_len = run_len_from(slice, idx, d0);
            if run_len >= 2 {
                if d0 <= u64::MAX as u128 {
                    buf.push(TAG_RUN_U64);
                    buf.extend_from_slice(&(run_len as u32).to_le_bytes());
                    buf.extend_from_slice(&(d0 as u64).to_le_bytes());
                } else {
                    buf.push(TAG_RUN_U128);
                    buf.extend_from_slice(&(run_len as u32).to_le_bytes());
                    buf.extend_from_slice(&d0.to_le_bytes());
                }
                idx += run_len;
            } else if d0 <= u64::MAX as u128 {
                let mut count = 1usize;
                while idx + count < slice.len() {
                    let d = delta_at(slice, idx + count);
                    if d > u64::MAX as u128 || run_len_from(slice, idx + count, d) >= 3 {
                        break;
                    }
                    count += 1;
                }
                if buf.len().saturating_add(5 + count * 8 + 2) >= raw_byte_len {
                    return None;
                }
                buf.push(TAG_DELTAS_U64);
                buf.extend_from_slice(&(count as u32).to_le_bytes());
                for j in idx..idx + count {
                    buf.extend_from_slice(&(delta_at(slice, j) as u64).to_le_bytes());
                }
                idx += count;
            } else {
                let mut count = 1usize;
                while idx + count < slice.len() {
                    let d = delta_at(slice, idx + count);
                    if d <= u64::MAX as u128 || run_len_from(slice, idx + count, d) >= 2 {
                        break;
                    }
                    count += 1;
                }
                if buf.len().saturating_add(5 + count * 16 + 2) >= raw_byte_len {
                    return None;
                }
                buf.push(TAG_DELTAS_U128);
                buf.extend_from_slice(&(count as u32).to_le_bytes());
                for j in idx..idx + count {
                    buf.extend_from_slice(&delta_at(slice, j).to_le_bytes());
                }
                idx += count;
            }

            if buf.len() + 2 >= raw_byte_len {
                return None;
            }
        }

        if buf.len() % 16 == 0 {
            buf.push(0);
        }
        if buf.len() < raw_byte_len {
            Some(buf)
        } else {
            None
        }
    }

    pub(crate) struct EntitySliceRef<'a>(pub(crate) &'a [EntityId]);

    impl Serialize for EntitySliceRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if serializer.is_human_readable() {
                self.0.serialize(serializer)
            } else if let Some(compressed) = encode_delta_rle(self.0) {
                serializer.serialize_bytes(&compressed)
            } else {
                // SAFETY: EntityId is #[repr(transparent)] over Uuid ([u8; 16]),
                // with size 16, alignment 1, and no padding bytes.
                let byte_len = self
                    .0
                    .len()
                    .checked_mul(16)
                    .ok_or_else(|| serde::ser::Error::custom("EntityId slice byte length overflow"))?;
                let raw_bytes = unsafe {
                    std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), byte_len)
                };
                serializer.serialize_bytes(raw_bytes)
            }
        }
    }

    pub(crate) fn serialize<S: Serializer>(
        entities: &[EntityId],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        EntitySliceRef(entities).serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<EntityId>, D::Error> {
        if deserializer.is_human_readable() {
            Vec::<EntityId>::deserialize(deserializer)
        } else {
            struct EntityVecVisitor;
            impl<'de> serde::de::Visitor<'de> for EntityVecVisitor {
                type Value = Vec<EntityId>;

                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("a 16-byte-aligned or Delta-RLE compressed EntityId byte buffer")
                }

                fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                    if v.len() % 16 == 0 {
                        let count = v.len() / 16;
                        let mut out = Vec::<EntityId>::new();
                        out.try_reserve_exact(count)
                            .map_err(|_| E::custom("Delta-RLE entities allocation failed"))?;
                        if count > 0 {
                            // SAFETY: `EntityId` is 16 bytes with alignment 1 and any 16-byte
                            // pattern is a valid `Uuid` / `EntityId`. `out` has capacity `count`.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    v.as_ptr(),
                                    out.as_mut_ptr().cast::<u8>(),
                                    v.len(),
                                );
                                out.set_len(count);
                            }
                        }
                        return Ok(out);
                    }

                    if v.len() >= 34 && v[0] == DELTA_RLE_MAGIC {
                        let total_count =
                            u32::from_le_bytes(v[1..5].try_into().unwrap()) as usize;
                        if total_count < 4 {
                            return Err(E::custom("Invalid Delta-RLE entity count"));
                        }
                        let mut out = Vec::<EntityId>::new();
                        out.try_reserve_exact(total_count)
                            .map_err(|_| E::custom("Delta-RLE entities allocation failed"))?;
                        let mut current = u128::from_le_bytes(v[5..21].try_into().unwrap());
                        out.push(EntityId::from_u128(current));
                        let mut pos = 21usize;
                        while out.len() < total_count {
                            if pos + 5 > v.len() {
                                return Err(E::custom("Truncated Delta-RLE entity segment header"));
                            }
                            let tag = v[pos];
                            let count =
                                u32::from_le_bytes(v[pos + 1..pos + 5].try_into().unwrap())
                                    as usize;
                            pos += 5;
                            if count == 0
                                || out
                                    .len()
                                    .checked_add(count)
                                    .is_none_or(|n| n > total_count)
                            {
                                return Err(E::custom("Invalid Delta-RLE entity segment count"));
                            }
                            match tag {
                                TAG_RUN_U64 => {
                                    if pos + 8 > v.len() {
                                        return Err(E::custom("Truncated Delta-RLE u64 run"));
                                    }
                                    let delta =
                                        u64::from_le_bytes(v[pos..pos + 8].try_into().unwrap())
                                            as u128;
                                    pos += 8;
                                    for _ in 0..count {
                                        current = current.wrapping_add(delta);
                                        out.push(EntityId::from_u128(current));
                                    }
                                }
                                TAG_RUN_U128 => {
                                    if pos + 16 > v.len() {
                                        return Err(E::custom("Truncated Delta-RLE u128 run"));
                                    }
                                    let delta =
                                        u128::from_le_bytes(v[pos..pos + 16].try_into().unwrap());
                                    pos += 16;
                                    for _ in 0..count {
                                        current = current.wrapping_add(delta);
                                        out.push(EntityId::from_u128(current));
                                    }
                                }
                                TAG_DELTAS_U64 => {
                                    let byte_len = count.checked_mul(8).ok_or_else(|| {
                                        E::custom("Delta-RLE u64 segment overflow")
                                    })?;
                                    if pos + byte_len > v.len() {
                                        return Err(E::custom("Truncated Delta-RLE u64 deltas"));
                                    }
                                    for chunk in v[pos..pos + byte_len].chunks_exact(8) {
                                        let delta =
                                            u64::from_le_bytes(chunk.try_into().unwrap()) as u128;
                                        current = current.wrapping_add(delta);
                                        out.push(EntityId::from_u128(current));
                                    }
                                    pos += byte_len;
                                }
                                TAG_DELTAS_U128 => {
                                    let byte_len = count.checked_mul(16).ok_or_else(|| {
                                        E::custom("Delta-RLE u128 segment overflow")
                                    })?;
                                    if pos + byte_len > v.len() {
                                        return Err(E::custom("Truncated Delta-RLE u128 deltas"));
                                    }
                                    for chunk in v[pos..pos + byte_len].chunks_exact(16) {
                                        let delta =
                                            u128::from_le_bytes(chunk.try_into().unwrap());
                                        current = current.wrapping_add(delta);
                                        out.push(EntityId::from_u128(current));
                                    }
                                    pos += byte_len;
                                }
                                _ => {
                                    return Err(E::custom(
                                        "Invalid Delta-RLE entity compression tag",
                                    ));
                                }
                            }
                        }
                        let rem = v.len() - pos;
                        if rem > 1 || (rem == 1 && v[pos] != 0) {
                            return Err(E::custom("Trailing bytes in Delta-RLE entity stream"));
                        }
                        return Ok(out);
                    }

                    Err(E::custom(
                        "Delta-RLE entities byte length is not a multiple of 16",
                    ))
                }

                fn visit_byte_buf<E: serde::de::Error>(
                    self,
                    v: Vec<u8>,
                ) -> Result<Self::Value, E> {
                    self.visit_bytes(&v)
                }

                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> Result<Self::Value, A::Error> {
                    let mut bytes =
                        Vec::<u8>::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
                    while let Some(b) = seq.next_element()? {
                        bytes.push(b);
                    }
                    self.visit_bytes(&bytes)
                }
            }
            deserializer.deserialize_bytes(EntityVecVisitor)
        }
    }
}

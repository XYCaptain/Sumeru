use super::*;

/// 强类型连续列迭代器
pub enum ColumnIter<'a, T: ModelComponent + Serialize + DeserializeOwned> {
    Direct(std::slice::Iter<'a, T>),
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> Clone for ColumnIter<'a, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Direct(iter) => Self::Direct(iter.clone()),
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> Iterator for ColumnIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Direct(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> ExactSizeIterator for ColumnIter<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Direct(iter) => iter.len(),
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> std::iter::FusedIterator
    for ColumnIter<'a, T>
{
}

use eyre::{eyre, Result};
use std::collections::HashMap;
use std::path::Path;

/// Every f32 tensor of a safetensors file, by name.
pub struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let file = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| eyre!("{e}"))?;
        let mut tensors = HashMap::new();
        for (name, view) in file.tensors() {
            eyre::ensure!(
                view.dtype() == safetensors::Dtype::F32,
                "{name}: expected f32, got {:?}",
                view.dtype()
            );
            let data = view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            tensors.insert(name, (view.shape().to_vec(), data));
        }
        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.entry(name)?.1.clone())
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.entry(name)?.0)
    }

    fn entry(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>)> {
        self.tensors
            .get(name)
            .ok_or_else(|| eyre!("missing weight {name}"))
    }
}

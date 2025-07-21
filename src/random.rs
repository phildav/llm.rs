use rand_mt::Mt;

pub fn random_permutation(data: &mut [usize], numel: usize, rng: &mut Mt) {
    for i in (1..numel).rev() {
        // pick an index j in [0, i] with equal probability
        let j = rng.next_u32() as usize %  (i + 1);
        // swap i <-> j
        let tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

pub fn init_identity_permutation(data: &mut [usize]) {
    for i in 0..data.len() {
        data[i] = i;
    }
}

#[cfg(test)]
mod tests {
    use rand_mt::Mt;

    #[test]
    fn mt19937_numerical_id() {
        // PyTorch/llm.c compatible seed
        let seed = 137u32;
        let mut rng = Mt::new(seed);

        // Integer values
        let expected = [
            4053805790u32,
            2173880614u32,
            380293709u32,
            1237255315u32,
            2986595568u32,
        ];

        for &exp in &expected {
            let val = rng.next_u32();
            assert_eq!(val, exp);
        }

    }
}
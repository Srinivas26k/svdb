use criterion::{black_box, criterion_group, criterion_main, Criterion};
use srvdb::{SvDB, VectorEngine, Vector};
use rand::Rng;

const DIM: usize = 1536;

fn generate_random_vector() -> Vector {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>() - 0.5).collect();
    Vector::new(data)
}

fn criterion_benchmark(c: &mut Criterion) {
    // 1. Setup DB
    let db_path = "./bench_db";
    let _ = std::fs::remove_dir_all(db_path); // Clean start
    let mut db = SvDB::new(db_path).unwrap();

    // 2. Populate with 10,000 vectors
    println!("Populating DB with 10,000 vectors for benchmark...");
    for i in 0..10_000 {
        let vec = generate_random_vector();
        db.add(&vec, &format!(r#"{{"id": {}}}"#, i)).unwrap();
    }
    db.persist().unwrap();
    println!("Database populated!");

    // 3. Define the Query
    let query = generate_random_vector();

    // 4. Run Benchmark
    c.bench_function("search_10k_vectors", |b| {
        b.iter(|| {
            // black_box prevents compiler from optimizing the code away
            db.search(black_box(&query), black_box(10)).unwrap();
        })
    });
    
    // Cleanup
    let _ = std::fs::remove_dir_all(db_path);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
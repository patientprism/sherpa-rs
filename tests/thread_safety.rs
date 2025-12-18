/// Thread safety stress test for EmbeddingExtractor
///
/// Run with a model file:
/// ```
/// SPEAKER_MODEL=path/to/model.onnx cargo test test_concurrent_embedding_extraction --release -- --nocapture
/// ```
///
/// To download a test model:
/// ```
/// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
/// ```
use std::sync::Arc;
use std::thread;

/// Test that multiple threads can safely use the same EmbeddingExtractor concurrently.
/// This verifies that changing `&mut self` to `&self` is safe.
#[test]
fn test_concurrent_embedding_extraction() {
    // Skip if no model is available
    let model_path = match std::env::var("SPEAKER_MODEL") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping test: SPEAKER_MODEL env var not set");
            eprintln!("Set SPEAKER_MODEL=/path/to/model.onnx to run this test");
            return;
        }
    };

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping test: model file not found at {}", model_path);
        return;
    }

    let config = sherpa_rs::speaker_id::ExtractorConfig {
        model: model_path,
        num_threads: Some(1), // Keep internal parallelism low to test external concurrency
        ..Default::default()
    };

    let extractor = Arc::new(
        sherpa_rs::speaker_id::EmbeddingExtractor::new(config)
            .expect("Failed to create extractor"),
    );

    // Generate test audio: 1 second of a simple sine wave at 440Hz
    let sample_rate = 16000u32;
    let duration_secs = 1.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();

    let num_threads = 8;
    let iterations_per_thread = 10;

    println!(
        "Running {} threads with {} iterations each",
        num_threads, iterations_per_thread
    );

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let extractor = Arc::clone(&extractor);
            let samples = samples.clone();

            thread::spawn(move || {
                for iter in 0..iterations_per_thread {
                    let result = extractor.compute_speaker_embedding(samples.clone(), sample_rate);

                    match result {
                        Ok(embedding) => {
                            // Verify embedding has expected size
                            assert_eq!(
                                embedding.len(),
                                extractor.embedding_size,
                                "Thread {} iter {}: embedding size mismatch",
                                thread_id,
                                iter
                            );
                            // Verify embedding contains valid floats
                            assert!(
                                embedding.iter().all(|v| v.is_finite()),
                                "Thread {} iter {}: embedding contains non-finite values",
                                thread_id,
                                iter
                            );
                        }
                        Err(e) => {
                            panic!("Thread {} iter {} failed: {}", thread_id, iter, e);
                        }
                    }
                }
                println!("Thread {} completed {} iterations", thread_id, iterations_per_thread);
            })
        })
        .collect();

    // Wait for all threads to complete
    for (i, handle) in handles.into_iter().enumerate() {
        handle.join().unwrap_or_else(|e| panic!("Thread {} panicked: {:?}", i, e));
    }

    println!(
        "All {} threads completed successfully ({} total inferences)",
        num_threads,
        num_threads * iterations_per_thread
    );
}

/// Test that Arc<EmbeddingExtractor> can be sent across threads (Send + Sync)
#[test]
fn test_extractor_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<sherpa_rs::speaker_id::EmbeddingExtractor>();
    assert_sync::<sherpa_rs::speaker_id::EmbeddingExtractor>();
    assert_send::<Arc<sherpa_rs::speaker_id::EmbeddingExtractor>>();
    assert_sync::<Arc<sherpa_rs::speaker_id::EmbeddingExtractor>>();
}

/// Test that Arc<Diarize> can be sent across threads (Send + Sync)
#[test]
fn test_diarize_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<sherpa_rs::diarize::Diarize>();
    assert_sync::<sherpa_rs::diarize::Diarize>();
    assert_send::<Arc<sherpa_rs::diarize::Diarize>>();
    assert_sync::<Arc<sherpa_rs::diarize::Diarize>>();
}

/// Thread safety stress test for Diarize
///
/// Run with model files:
/// ```
/// SEGMENTATION_MODEL=path/to/segmentation.onnx EMBEDDING_MODEL=path/to/embedding.onnx \
///   cargo test test_concurrent_diarization --release -- --nocapture
/// ```
#[test]
fn test_concurrent_diarization() {
    let segmentation_model = match std::env::var("SEGMENTATION_MODEL") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping test: SEGMENTATION_MODEL env var not set");
            return;
        }
    };

    let embedding_model = match std::env::var("EMBEDDING_MODEL") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping test: EMBEDDING_MODEL env var not set");
            return;
        }
    };

    if !std::path::Path::new(&segmentation_model).exists() {
        eprintln!("Skipping test: segmentation model not found at {}", segmentation_model);
        return;
    }
    if !std::path::Path::new(&embedding_model).exists() {
        eprintln!("Skipping test: embedding model not found at {}", embedding_model);
        return;
    }

    let config = sherpa_rs::diarize::DiarizeConfig {
        num_embedding_model_threads: Some(1),
        num_segmentation_model_threads: Some(1),
        ..Default::default()
    };

    let diarizer = Arc::new(
        sherpa_rs::diarize::Diarize::new(&segmentation_model, &embedding_model, config)
            .expect("Failed to create diarizer"),
    );

    // Generate test audio: 3 seconds of audio (diarization needs longer samples)
    let sample_rate = 16000u32;
    let duration_secs = 3.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3
        })
        .collect();

    let num_threads = 4;
    let iterations_per_thread = 3;

    println!(
        "Running {} threads with {} iterations each for diarization",
        num_threads, iterations_per_thread
    );

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let diarizer = Arc::clone(&diarizer);
            let samples = samples.clone();

            thread::spawn(move || {
                for iter in 0..iterations_per_thread {
                    let result = diarizer.compute(samples.clone(), None);

                    match result {
                        Ok(segments) => {
                            println!(
                                "Thread {} iter {}: got {} segments",
                                thread_id,
                                iter,
                                segments.len()
                            );
                        }
                        Err(e) => {
                            // Diarization might fail on synthetic audio, that's OK
                            // We're testing thread safety, not accuracy
                            println!("Thread {} iter {}: {}", thread_id, iter, e);
                        }
                    }
                }
                println!("Thread {} completed {} iterations", thread_id, iterations_per_thread);
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        handle.join().unwrap_or_else(|e| panic!("Thread {} panicked: {:?}", i, e));
    }

    println!(
        "All {} threads completed successfully ({} total diarizations)",
        num_threads,
        num_threads * iterations_per_thread
    );
}

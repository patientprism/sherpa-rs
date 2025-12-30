use crate::{get_default_provider, utils::RawCStr};
use eyre::{bail, Result};
use std::{
    collections::{btree_map, BTreeMap},
    path::Path,
};

#[derive(Debug)]
pub struct Diarize {
    sd: *const sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarization,
    extract_speaker_embeddings: bool,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub speaker: i32,
}

type ProgressCallback = Box<dyn Fn(i32, i32) -> i32 + Send + 'static>;

#[derive(Debug, Clone)]
pub struct DiarizeConfig {
    pub num_clusters: Option<i32>,
    pub threshold: Option<f32>,
    pub min_duration_on: Option<f32>,
    pub min_duration_off: Option<f32>,
    pub provider: Option<String>,
    pub num_embedding_model_threads: Option<usize>,
    pub num_segmentation_model_threads: Option<usize>,
    pub debug: bool,
    pub extract_speaker_embeddings: bool,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self {
            num_clusters: Some(4),
            threshold: Some(0.5),
            min_duration_on: Some(0.0),
            min_duration_off: Some(0.0),
            provider: None,
            num_embedding_model_threads: Some(1),
            num_segmentation_model_threads: Some(1),
            debug: false,
            extract_speaker_embeddings: false,
        }
    }
}

impl Diarize {
    pub fn new<P: AsRef<Path>>(
        segmentation_model: P,
        embedding_model: P,
        config: DiarizeConfig,
    ) -> Result<Self> {
        let provider = config.provider.unwrap_or(get_default_provider());

        let debug = config.debug;
        let debug = if debug { 1 } else { 0 };

        let embedding_model = embedding_model.as_ref().to_str().unwrap();
        let segmentation_model = segmentation_model.as_ref().to_str().unwrap();

        let clustering_config = sherpa_rs_sys::SherpaOnnxFastClusteringConfig {
            num_clusters: config.num_clusters.unwrap_or(4),
            threshold: config.threshold.unwrap_or(0.5),
        };

        let embedding_model = RawCStr::new(embedding_model);
        let provider = RawCStr::new(&provider.clone());
        let segmentation_model = RawCStr::new(segmentation_model);

        let config = sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationConfig {
            embedding: sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
                model: embedding_model.as_ptr(),
                num_threads: config.num_embedding_model_threads.unwrap_or(1) as _,
                debug,
                provider: provider.as_ptr(),
            },
            clustering: clustering_config,
            min_duration_off: config.min_duration_off.unwrap_or(0.0),
            min_duration_on: config.min_duration_on.unwrap_or(0.0),
            extract_speaker_embeddings: config.extract_speaker_embeddings,
            segmentation: sherpa_rs_sys::SherpaOnnxOfflineSpeakerSegmentationModelConfig {
                pyannote: sherpa_rs_sys::SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
                    model: segmentation_model.as_ptr(),
                },
                num_threads: config.num_segmentation_model_threads.unwrap_or(1) as _,
                debug,
                provider: provider.as_ptr(),
            },
        };

        let sd = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineSpeakerDiarization(&config) };
        if sd.is_null() {
            bail!("Failed to initialize offline speaker diarization")
        }

        Ok(Self {
            sd,
            extract_speaker_embeddings: config.extract_speaker_embeddings,
        })
    }

    pub fn compute(
        &mut self,
        mut samples: Vec<f32>,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<ComputedDiarizationSegments> {
        let samples_ptr = samples.as_mut_ptr();
        let mut segments = Vec::new();
        let mut speaker_embeddings = BTreeMap::new();
        unsafe {
            let mut callback_box =
                progress_callback.map(|cb| Box::new(cb) as Box<ProgressCallback>);
            let callback_ptr = callback_box
                .as_mut()
                .map(|b| b.as_mut() as *mut ProgressCallback as *mut std::ffi::c_void)
                .unwrap_or(std::ptr::null_mut());

            let result = sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                self.sd,
                samples_ptr,
                samples.len() as i32,
                if callback_box.is_some() {
                    Some(progress_callback_wrapper)
                } else {
                    None
                },
                callback_ptr,
            );

            let num_segments =
                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
            let segments_ptr: *const sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationSegment =
                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);

            if !segments_ptr.is_null() && num_segments > 0 {
                let segments_result: &[sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationSegment] =
                    std::slice::from_raw_parts(segments_ptr, num_segments as usize);

                for segment in segments_result {
                    segments.push(Segment {
                        start: segment.start,
                        end: segment.end,
                        speaker: segment.speaker,
                    });

                    if self.extract_speaker_embeddings {
                        if let btree_map::Entry::Vacant(v) =
                            speaker_embeddings.entry(segment.speaker)
                        {
                            let mut speaker_embeddings = std::ptr::null_mut();
                            let mut speaker_embeddings_len: i32 = 0;

                            sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultGetSpeakerEmbeddings(
                                result,
                                segment.speaker,
                                &mut speaker_embeddings,
                                &mut speaker_embeddings_len,
                            );

                            if !speaker_embeddings.is_null() {
                                v.insert(
                                    std::slice::from_raw_parts(
                                        speaker_embeddings,
                                        speaker_embeddings_len as usize,
                                    )
                                    .to_vec(),
                                );

                                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultFreeSpeakerEmbeddings(
                                    speaker_embeddings,
                                );
                            } else {
                                debug_assert!(
                                    false,
                                    "Speaker embeddings not found for speaker {}",
                                    segment.speaker
                                );
                            }
                        }
                    }
                }
            } else {
                bail!("No segments found or invalid pointer.");
            }

            sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments_ptr);
            sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);

            Ok(ComputedDiarizationSegments {
                segments,
                speaker_embeddings,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComputedDiarizationSegments {
    segments: Vec<Segment>,
    speaker_embeddings: BTreeMap<i32, Vec<f32>>,
}
impl IntoIterator for ComputedDiarizationSegments {
    type IntoIter = std::vec::IntoIter<Segment>;
    type Item = Segment;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.segments.into_iter()
    }
}
impl ComputedDiarizationSegments {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    #[inline(always)]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn speaker_embeddings(&self, speaker_label: i32) -> Option<&[f32]> {
        self.speaker_embeddings
            .get(&speaker_label)
            .map(|embeddings| &**embeddings)
    }
}

unsafe extern "C" fn progress_callback_wrapper(
    num_processed_chunk: i32,
    num_total_chunks: i32,
    arg: *mut std::ffi::c_void,
) -> i32 {
    let callback = &mut *(arg as *mut ProgressCallback);
    callback(num_processed_chunk, num_total_chunks)
}

unsafe impl Send for Diarize {}
unsafe impl Sync for Diarize {}

impl Drop for Diarize {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineSpeakerDiarization(self.sd);
        }
    }
}

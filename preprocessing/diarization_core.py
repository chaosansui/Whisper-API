import torch
import numpy as np
from speechbrain.inference import EncoderClassifier 
# 导入新的聚类库：Agglomerative Clustering 和 Silhouette Score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple
import os

logger = logging.getLogger(__name__)

class DiarizationCore:
    """负责提取说话人嵌入、执行聚类和分配说话人标签。"""
    
    # K 值搜索范围：对于客服场景，K=2 到 K=5 是合理的范围
    MIN_K_SPEAKERS = 2
    MAX_K_SPEAKERS = 5 
    # 最小片段时长：低于此值的片段不进行嵌入提取
    MIN_DURATION_FOR_EMBEDDING = 0.25 

    def __init__(self, speaker_embedding_path: str, sample_rate: int, device: str):
        self.sample_rate = sample_rate
        self.device = device
        self.speaker_embedding_path = speaker_embedding_path
        self.speaker_embedder = None
        self._load_embedder()

    def _load_embedder(self):
        """加载 SpeechBrain ECAPA-TDNN 模型 (在 GPU 上运行)"""
        if not os.path.isdir(self.speaker_embedding_path):
             raise ValueError(f"❌ SpeakerEmbedding 模型目录缺失: {self.speaker_embedding_path}")
        
        spk_run_options = {"device": self.device}
        logger.info(f"🔄 加载 SpeakerEmbedding 模型，使用设备: {self.device}")
        self.speaker_embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self.speaker_embedding_path,
            run_opts=spk_run_options
        )
        self.speaker_embedder.eval()
        self.speaker_embedder.to(self.device)
        logger.info("✅ SpeakerEmbedding 模型加载完成")

    def _extract_embeddings(self, audio_np: np.ndarray, segments: List[Dict]) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        [流程 8: 嵌入提取]
        从 VAD 片段中提取 ECAPA-TDNN 嵌入向量。
        """
        embeddings = []
        valid_segments = []
        min_samples = int(self.MIN_DURATION_FOR_EMBEDDING * self.sample_rate)

        for segment in segments:
            start, end = segment["start"], segment["end"]
            
            # 从音频 numpy 数组中裁剪片段
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            audio_segment = audio_np[start_sample:end_sample]
            
            # 过滤太短的片段 (小于 0.25s)
            if len(audio_segment) < min_samples: 
                continue 

            audio_segment_tensor = torch.from_numpy(audio_segment).float().to(self.device).unsqueeze(0)
            
            # 提取嵌入
            with torch.no_grad():
                # 关键：使用 normalize=True 确保嵌入向量是单位向量
                embedding = self.speaker_embedder.encode_batch(audio_segment_tensor, normalize=True)
            
            embeddings.append(embedding.cpu().numpy().squeeze())
            valid_segments.append(segment)
            
        return embeddings, valid_segments

    def _determine_optimal_k(self, embeddings_np: np.ndarray) -> int:
        """
        使用 Agglomerative Clustering 和 Silhouette Score 动态确定最优 K 值。
        """
        n_samples = len(embeddings_np)
        
        # 如果有效片段少于 2 个，则只能是 K=1
        if n_samples < self.MIN_K_SPEAKERS:
             return 1
             
        # K 值搜索范围上限
        max_k = min(n_samples - 1, self.MAX_K_SPEAKERS)
        if max_k < self.MIN_K_SPEAKERS:
             return 1 

        best_k = self.MIN_K_SPEAKERS
        best_score = -1.0
        
        logger.info(f"🔍 正在 {self.MIN_K_SPEAKERS} 到 {max_k} 范围内搜索最优 K 值...")

        # 注意：Agglomerative Clustering 需要原始嵌入，而不是相似度矩阵
        for k in range(self.MIN_K_SPEAKERS, max_k + 1):
            try:
                clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = clustering.fit_predict(embeddings_np)
                
                # 计算 Silhouette Score (轮廓系数)
                score = silhouette_score(embeddings_np, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                
                logger.debug(f"  K={k}, Silhouette Score: {score:.4f}")
            except Exception as e:
                # 聚类可能在某些极端情况下失败
                logger.warning(f"K={k} 聚类计算失败: {e}")
                continue

        logger.info(f"🎯 选定最优 K={best_k} (Score: {best_score:.4f})")
        return best_k


    def _cluster_and_label(self, embeddings: List[np.ndarray], segments: List[Dict]) -> List[Dict]:
        """
        [流程 5: 说话人分割] 和 [流程 9: 聚类合并] 的聚类部分
        执行 Agglomerative Clustering 并分配说话人标签。
        """
        if len(embeddings) < 2:
            logger.warning("❌ 提取的有效片段不足 2 个，跳过聚类。")
            return [{**s, "speaker": "SPEAKER_00"} for s in segments] 

        logger.info("🎯 开始说话人聚类 (Agglomerative Clustering)...")
        
        embeddings_np = np.array(embeddings)
        
        # 1. 动态确定 K 值
        K_ESTIMATE = self._determine_optimal_k(embeddings_np)
        
        # 2. 使用确定的 K 值进行最终聚类
        if K_ESTIMATE == 1:
            labels = np.zeros(len(embeddings_np), dtype=int)
        else:
            final_clustering = AgglomerativeClustering(
                n_clusters=K_ESTIMATE, 
                linkage='ward'
            )
            labels = final_clustering.fit_predict(embeddings_np)

        # 分配说话人标签
        for i, segment in enumerate(segments):
            segment["speaker"] = f"SPEAKER_{labels[i]:02d}"
        
        return segments


    def diarize(self, audio_np: np.ndarray, vad_segments: List[Dict]) -> List[Dict]:
        """
        执行完整的 Diarization 流程：提取嵌入、聚类、分配标签。
        """
        if not vad_segments:
            logger.warning("Diadization Core 接收到空片段，跳过处理。")
            return []

        embeddings, valid_segments = self._extract_embeddings(audio_np, vad_segments)
        
        if not valid_segments:
            logger.warning("所有 VAD 片段均被过滤，无法提取嵌入。")
            return []
            
        labeled_segments = self._cluster_and_label(embeddings, valid_segments)
        
        # 3. 合并相邻的相同说话人片段 (流程 9/10 的一部分)
        merged_segments = self._merge_speaker_segments(labeled_segments)
        logger.info(f"🎯 Diarization Core 输出 {len(merged_segments)} 个合并片段")

        return merged_segments
        
    def _merge_speaker_segments(self, segments: List[Dict]) -> List[Dict]:
        """合并相同说话人的相邻片段"""
        if not segments:
            return []
        
        merged_segments = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # 说话人相同且间隔小于 1.5 秒则合并
            if (current_segment['speaker'] == next_segment['speaker'] and 
                next_segment['start'] - current_segment['end'] < 1.5):
                current_segment['end'] = next_segment['end']
            else:
                if current_segment['end'] - current_segment['start'] >= 0.1:
                    merged_segments.append(current_segment)
                current_segment = next_segment
        
        if current_segment['end'] - current_segment['start'] >= 0.1:
            merged_segments.append(current_segment)
        
        return merged_segments
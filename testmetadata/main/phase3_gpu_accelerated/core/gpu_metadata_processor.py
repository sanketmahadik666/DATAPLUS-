"""
Phase 3: GPU-Accelerated Metadata Processor

Enhanced metadata extraction with GPU acceleration for real-time document processing.
Supports deep feature extraction and optimized parallel processing.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# GPU/ML Libraries
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, efficientnet_b0
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndimage
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import cv2
    import cv2.cuda as cuda_cv
    HAS_OPENCV_CUDA = True
except ImportError:
    HAS_OPENCV_CUDA = False
    cv2 = None

# Document Processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

logger = logging.getLogger(__name__)


class GPUFeatureExtractor:
    """GPU-accelerated feature extraction using deep learning models."""

    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.transforms = {}
        self._load_models()

    def _setup_device(self, device: str):
        """Setup GPU/CPU device configuration."""
        if not HAS_TORCH:
            raise ImportError("PyTorch not available. Install torch>=2.0.0")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if device == "cuda":
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        logger.info(f"Using device: {self.device}")
        return self.device

    def _load_models(self):
        """Load pre-trained deep learning models for feature extraction."""
        if not HAS_TORCH:
            return

        # ResNet50 for general image features
        resnet = resnet50(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove classifier
        resnet.to(self.device)
        resnet.eval()
        self.models['resnet50'] = resnet

        # EfficientNet for efficient feature extraction
        efficientnet = efficientnet_b0(pretrained=True)
        efficientnet = nn.Sequential(*list(efficientnet.children())[:-1])
        efficientnet.to(self.device)
        efficientnet.eval()
        self.models['efficientnet'] = efficientnet

        # Image transforms
        self.transforms['resnet'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transforms['efficientnet'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_deep_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract deep features using GPU-accelerated models."""
        if not HAS_TORCH or not self.models:
            return {}

        features = {}

        try:
            # ResNet50 features
            with torch.no_grad():
                tensor = self.transforms['resnet'](image).unsqueeze(0).to(self.device)
                resnet_features = self.models['resnet50'](tensor)
                features['resnet50'] = resnet_features.squeeze().cpu().numpy()

            # EfficientNet features
            with torch.no_grad():
                tensor = self.transforms['efficientnet'](image).unsqueeze(0).to(self.device)
                eff_features = self.models['efficientnet'](tensor)
                features['efficientnet'] = eff_features.squeeze().cpu().numpy()

        except Exception as e:
            logger.warning(f"Deep feature extraction failed: {e}")

        return features


class GPUImageProcessor:
    """GPU-accelerated image processing operations."""

    def __init__(self):
        self.use_cuda = HAS_CUPY or (HAS_OPENCV_CUDA and cv2 and cv2.cuda.getCudaEnabledDeviceCount() > 0)

    def preprocess_image_gpu(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """GPU-accelerated image preprocessing."""
        if not self.use_cuda:
            return self._cpu_preprocessing(image)

        try:
            if HAS_CUPY:
                return self._cupy_preprocessing(image)
            elif HAS_OPENCV_CUDA:
                return self._opencv_cuda_preprocessing(image)
        except Exception as e:
            logger.warning(f"GPU preprocessing failed, falling back to CPU: {e}")
            return self._cpu_preprocessing(image)

    def _cupy_preprocessing(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """CuPy-based GPU preprocessing."""
        gpu_image = cp.asarray(image)

        # Convert to grayscale if needed
        if len(gpu_image.shape) == 3:
            gpu_image = cp.dot(gpu_image[..., :3], cp.array([0.2989, 0.5870, 0.1140]))

        # Denoise using Gaussian filter
        denoised = cndimage.gaussian_filter(gpu_image.astype(cp.float32), sigma=1.0)

        # Enhance contrast
        min_val, max_val = cp.min(denoised), cp.max(denoised)
        enhanced = cp.clip((denoised - min_val) / (max_val - min_val) * 255, 0, 255).astype(cp.uint8)

        processed = cp.asnumpy(enhanced)

        metadata = {
            'processing_method': 'cupy_gpu',
            'gpu_accelerated': True,
            'filters_applied': ['grayscale', 'denoise', 'contrast_enhancement']
        }

        return processed, metadata

    def _opencv_cuda_preprocessing(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """OpenCV CUDA-based preprocessing."""
        gpu_image = cuda_cv.GpuMat()
        gpu_image.upload(image)

        # Convert to grayscale
        gpu_gray = cuda_cv.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)

        # Denoise
        gpu_denoised = cuda_cv.GaussianBlur(gpu_gray, (5, 5), 1.0)

        # Download back to CPU
        processed = gpu_denoised.download()

        metadata = {
            'processing_method': 'opencv_cuda',
            'gpu_accelerated': True,
            'filters_applied': ['grayscale', 'gaussian_blur']
        }

        return processed, metadata

    def _cpu_preprocessing(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """CPU fallback preprocessing."""
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if cv2 else image.mean(axis=2).astype(np.uint8)
        else:
            processed = image

        if cv2:
            processed = cv2.GaussianBlur(processed, (5, 5), 1.0)

        metadata = {
            'processing_method': 'cpu',
            'gpu_accelerated': False,
            'filters_applied': ['grayscale', 'gaussian_blur'] if cv2 else ['basic']
        }

        return processed, metadata


class GPUMetadataProcessor:
    """Main GPU-accelerated metadata processor combining all components."""

    def __init__(self, max_workers: int = 16, device: str = "auto"):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize components
        self.feature_extractor = GPUFeatureExtractor(device) if HAS_TORCH else None
        self.image_processor = GPUImageProcessor()

        # GPU memory management
        self.gpu_memory_limit = 0.8  # Use 80% of GPU memory
        self._setup_gpu_memory_management()

        logger.info(f"GPU Metadata Processor initialized with {max_workers} workers")

    def _setup_gpu_memory_management(self):
        """Setup GPU memory management for efficient processing."""
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_limit)
            torch.cuda.empty_cache()

    async def process_document_batch(self, pdf_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of PDF documents with GPU acceleration."""
        start_time = time.time()

        # Create processing tasks
        tasks = []
        for pdf_path in pdf_paths:
            task = asyncio.get_event_loop().run_in_executor(
                self.executor, self._process_single_document, pdf_path
            )
            tasks.append(task)

        # Execute batch processing
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {pdf_paths[i].name}: {result}")
                processed_results.append(self._create_error_result(pdf_paths[i], str(result)))
            else:
                processed_results.append(result)

        processing_time = time.time() - start_time
        files_per_second = len(pdf_paths) / processing_time if processing_time > 0 else 0

        logger.info(".2f")

        return processed_results

    def _process_single_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF document."""
        try:
            # Extract images from PDF
            images = self._extract_pdf_images(pdf_path)
            if not images:
                return self._create_minimal_result(pdf_path)

            # Process first page image
            primary_image = images[0]
            processed_image, processing_metadata = self.image_processor.preprocess_image_gpu(primary_image)

            # Extract features
            features = self._extract_comprehensive_features(processed_image, primary_image)

            # Deep features (if GPU available)
            deep_features = {}
            if self.feature_extractor:
                deep_features = self.feature_extractor.extract_deep_features(primary_image)

            # Combine all metadata
            result = {
                'document_id': pdf_path.stem,
                'file_path': str(pdf_path),
                'file_size': pdf_path.stat().st_size,
                'processing_status': 'success',
                'gpu_accelerated': self.image_processor.use_cuda,

                # Image processing metadata
                'pages_processed': len(images),
                'image_dimensions': primary_image.shape[:2] if len(primary_image.shape) >= 2 else None,

                # Processing metadata
                'processing_method': processing_metadata.get('processing_method', 'unknown'),
                'filters_applied': processing_metadata.get('filters_applied', []),

                # Features
                **features,

                # Deep features
                'deep_features': deep_features,

                # Performance metrics
                'processing_timestamp': time.time(),
            }

            return result

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            return self._create_error_result(pdf_path, str(e))

    def _extract_pdf_images(self, pdf_path: Path) -> List[np.ndarray]:
        """Extract images from PDF pages."""
        images = []

        try:
            if HAS_PYMUPDF:
                doc = fitz.open(pdf_path)
                max_pages = min(3, len(doc))  # Process up to 3 pages

                for page_num in range(max_pages):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality

                    # Convert to numpy array
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                    img_array = img_data.reshape(pix.height, pix.width, pix.n)

                    # Convert to RGB if necessary
                    if pix.n == 4:  # CMYK or RGBA
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB) if cv2 else img_array[..., :3]

                    images.append(img_array)

                doc.close()

            elif HAS_PDF2IMAGE:
                pil_images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=3)
                for pil_img in pil_images:
                    images.append(np.array(pil_img))

        except Exception as e:
            logger.warning(f"PDF image extraction failed for {pdf_path.name}: {e}")

        return images

    def _extract_comprehensive_features(self, processed_image: np.ndarray, original_image: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive features from processed images."""
        features = {}

        try:
            # Basic image features
            height, width = processed_image.shape[:2]
            features['image_height'] = height
            features['image_width'] = width
            features['aspect_ratio'] = width / height if height > 0 else 0

            # Statistical features
            features['mean_intensity'] = float(np.mean(processed_image))
            features['std_intensity'] = float(np.std(processed_image))
            features['min_intensity'] = float(np.min(processed_image))
            features['max_intensity'] = float(np.max(processed_image))

            if cv2:
                # Edge detection features
                edges = cv2.Canny(processed_image, 100, 200)
                features['edge_density'] = float(np.mean(edges > 0))

                # Texture features using GLCM-like approach
                features['contrast'] = float(self._calculate_contrast(processed_image))
                features['entropy'] = float(self._calculate_entropy(processed_image))

            # Color features (if applicable)
            if len(original_image.shape) == 3:
                features['color_channels'] = original_image.shape[2]
                features['is_color'] = original_image.shape[2] > 1

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")

        return features

    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        return float(np.std(image))

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy."""
        histogram = np.histogram(image, bins=256, range=(0, 256))[0]
        histogram = histogram / histogram.sum()
        histogram = histogram[histogram > 0]
        return -np.sum(histogram * np.log2(histogram))

    def _create_minimal_result(self, pdf_path: Path) -> Dict[str, Any]:
        """Create minimal result for failed PDF processing."""
        return {
            'document_id': pdf_path.stem,
            'file_path': str(pdf_path),
            'processing_status': 'partial',
            'gpu_accelerated': False,
            'error': 'Could not extract images from PDF',
            'file_size': pdf_path.stat().st_size,
            'processing_timestamp': time.time(),
        }

    def _create_error_result(self, pdf_path: Path, error: str) -> Dict[str, Any]:
        """Create error result for failed processing."""
        return {
            'document_id': pdf_path.stem,
            'file_path': str(pdf_path),
            'processing_status': 'error',
            'gpu_accelerated': False,
            'error': error,
            'file_size': pdf_path.stat().st_size if pdf_path.exists() else None,
            'processing_timestamp': time.time(),
        }

    def cleanup(self):
        """Cleanup resources and GPU memory."""
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.executor.shutdown(wait=True)
        logger.info("GPU Metadata Processor cleaned up")


# Convenience function for easy usage
async def process_documents_gpu(pdf_paths: List[Path], max_workers: int = 16) -> List[Dict[str, Any]]:
    """Convenience function to process documents with GPU acceleration."""
    processor = GPUMetadataProcessor(max_workers=max_workers)
    try:
        results = await processor.process_document_batch(pdf_paths)
        return results
    finally:
        processor.cleanup()
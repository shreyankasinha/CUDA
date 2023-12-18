#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

__global__ void FloatToChar(float * input, unsigned char * gsoutput, unsigned char * charoutput, int size)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = tx + bx * BLOCK_SIZE;
  if (index >= size) return;
  
  float r = input[3 * index];
  float g = input[3 * index + 1];
  float b = input[3 * index + 2];
  unsigned char ri = (unsigned char) (r * 255.0);
  unsigned char gi = (unsigned char) (g * 255.0);
  unsigned char bi = (unsigned char) (b * 255.0);
  
  charoutput[3 * index] = ri;
  charoutput[3 * index + 1] = gi;
  charoutput[3 * index + 2] = bi;
  unsigned char tmp = (unsigned char) (0.21 * ri + 0.71 * gi + 0.07 * bi);
  gsoutput[index] = tmp;
}

__global__ void AtmcAdd(unsigned char * input_gray_image, unsigned int * output_bucket, int size)
{
  __shared__ unsigned int privateBucket[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = tx + bx * BLOCK_SIZE;
  int stride = BLOCK_SIZE;
  int idx = index;
  privateBucket[tx] = 0;
  __syncthreads();

  while (idx < size)
  {
    atomicAdd(&privateBucket[input_gray_image[idx]], 1);
    idx+=stride;
  }
  __syncthreads();
  
  output_bucket[tx] = privateBucket[tx];
}

__global__ void kernel_correct(unsigned char * input_image, float * corrected_cdf, float * outputImage, int size)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  unsigned int index = tx + bx * BLOCK_SIZE;
  if (index < size)
  {
    unsigned int startIdx = index * 3;
    for (int i = 0; i < 3; ++i)
    {
      outputImage[startIdx] = corrected_cdf[input_image[startIdx]];
      ++startIdx;
    }
  }
  __syncthreads();
}

void scan(unsigned int * input, float * histogram_cdf, unsigned int outputImageSize){
  unsigned int cum = 0;
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
  {
    cum += input[i];
    histogram_cdf[i] = (float) cum / outputImageSize;
  }
  return;
}

void correct(float min, float max, float * histogram_cdf)
{
  float range = max - min; 
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
  {
    histogram_cdf[i] = ((histogram_cdf[i] - min) / range);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  
  args = wbArg_read(argc, argv); 

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
 
  hostInputImageData = wbImage_getData(inputImage); 
  hostOutputImageData = wbImage_getData(outputImage); 
  wbTime_stop(Generic, "Importing data and creating memory on host");

  
  float * deviceInputImage;
  unsigned char * deviceOutputImage_uchar;
  unsigned char * deviceOutputImage_gray;
  unsigned int * deviceOutputBucket;
  unsigned int * histogram;
  float * histogram_cdf;
  float * device_histogram_cdf;
  float * device_final_image;
  
  
  unsigned int inputImageSize = imageWidth * imageHeight * imageChannels;
  unsigned int outputImageSize = imageWidth * imageHeight;
  cudaMalloc((void **) &deviceInputImage, sizeof(float) * inputImageSize);
  cudaMalloc((void **) &deviceOutputImage_uchar, sizeof(unsigned char) * inputImageSize);
  cudaMalloc((void **) &deviceOutputImage_gray, sizeof(unsigned char) * outputImageSize);
  cudaMalloc((void **) &deviceOutputBucket, sizeof(unsigned int) * HISTOGRAM_LENGTH);
  histogram = (unsigned int *) malloc(sizeof(unsigned int) * HISTOGRAM_LENGTH);
  histogram_cdf = (float *) malloc(sizeof(float) * HISTOGRAM_LENGTH);
  cudaMalloc((void **) &device_histogram_cdf, sizeof(float) * HISTOGRAM_LENGTH); 
  cudaMalloc((void **) &device_final_image, sizeof(float) * inputImageSize); 
  cudaMemcpy(deviceInputImage, hostInputImageData, sizeof(float) * inputImageSize, cudaMemcpyHostToDevice);
  int blockNumber = ceil((double) outputImageSize / BLOCK_SIZE);
  
  dim3 GridDim(blockNumber, 1, 1);
  dim3 BlockDim(BLOCK_SIZE, 1, 1);
  FloatToChar<<<GridDim, BlockDim>>>(deviceInputImage, deviceOutputImage_gray, deviceOutputImage_uchar, outputImageSize); 
  cudaDeviceSynchronize();
  
  dim3 GridDim_1(1, 1, 1);
  AtmcAdd<<<GridDim_1, BlockDim>>>(deviceOutputImage_gray, deviceOutputBucket, outputImageSize);
  cudaDeviceSynchronize();
  cudaMemcpy(histogram, deviceOutputBucket, sizeof(unsigned int) * HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost);
  
  scan(histogram, histogram_cdf, outputImageSize);
  float cdf_min = histogram_cdf[0];
  float cdf_max = 1.0;
  correct(cdf_min, cdf_max, histogram_cdf);
  

  cudaMemcpy(device_histogram_cdf, histogram_cdf, sizeof(float) * HISTOGRAM_LENGTH, cudaMemcpyHostToDevice);
  kernel_correct<<<GridDim, BlockDim>>>(deviceOutputImage_uchar, device_histogram_cdf, device_final_image, outputImageSize);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, device_final_image, sizeof(float) * inputImageSize, cudaMemcpyDeviceToHost);
  
  wbImage_setData(outputImage, hostOutputImageData);
  cudaFree(deviceInputImage);
  cudaFree(deviceOutputImage_uchar);
  cudaFree(deviceOutputImage_gray);
  cudaFree(deviceOutputBucket);
  cudaFree(device_histogram_cdf);
  cudaFree(device_final_image);
  
  wbSolution(args, outputImage);

  free(histogram);
  free(histogram_cdf);

  return 0;
}


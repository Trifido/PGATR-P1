//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
	system("pause");
    exit(1);
  }
}

//Voy a usar como variables constantes para los kernels, la matriz input de imagen y la matriz de filtro

#define TAMFILTRO 5

__constant__ float d_const_filter[TAMFILTRO*TAMFILTRO];

__global__
void box_filter(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols, const int filterWidth)
{
	const int2 thread_2D_pos = make_int2(blockIdx.y * blockDim.y + threadIdx.y,
		blockIdx.x * blockDim.x + threadIdx.x);
	const int thread_1D_pos = thread_2D_pos.x * numCols + thread_2D_pos.y;

	if (thread_2D_pos.x >= numRows || thread_2D_pos.y >= numCols)
		return;
	
	
	int contador = 0;
	float result = 0.0f;
	for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; ++filter_r){
		
		for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; ++filter_c){

			int image_r = thread_2D_pos.x + filter_r;
			int image_c = thread_2D_pos.y + filter_c;

			if ((image_c >= 0) && (image_c < numCols) && (image_r >= 0) && (image_r < numRows)){

				float image_value = inputChannel[image_r * numCols + image_c];
				float filter_value = d_const_filter[contador];
				result += image_value * filter_value;
			}
			contador++;
		}
	}
	

	outputChannel[thread_1D_pos] = result;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	int id = thread_1D_pos;
	redChannel[id] = inputImageRGBA[id].x;
	greenChannel[id] = inputImageRGBA[id].y;
	blueChannel[id] = inputImageRGBA[id].z;
}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
  // Copiar el filtro  (h_filter) a memoria global de la GPU (d_filter)

 // checkCudaErrors(cudaMalloc(&d_filter, sizeof(unsigned char) * filterWidth * filterWidth));
  //checkCudaErrors(cudaMemcpyToSymbol(d_const_filter, h_filter, sizeof(unsigned char) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemcpyToSymbol(d_const_filter, h_filter, sizeof(float) * filterWidth * filterWidth));
 // cudaMemcpy(d_filter, h_filter, sizeof(unsigned char) * filterWidth * filterWidth, cudaMemcpyHostToDevice);//Copiamos el d_filter a GPU.

}


void create_filter(float **h_filter, int *filterWidth){

  const int KernelWidth = 5; //OJO CON EL TAMA�O DEL FILTRO//
  *filterWidth = KernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[KernelWidth * KernelWidth];
  
  /*
  //Filtro gaussiano: blur
  const float KernelSigma = 2.;

  float filterSum = 0.f; //for normalization

  for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
    for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
      (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
    for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
      (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
    }
  }
  */

  //Laplaciano 5x5
  /*(*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
  (*h_filter)[5] = 1.;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
  (*h_filter)[10] = -1.;(*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
  (*h_filter)[15] = 1.; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
  (*h_filter)[20] = 0; (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;*/
  
  //ESTOS DOS M�TODOS LOS HE SACADO DE INTERNET Y DAN UN RESULTADO PARECIDO. Se aprecia la eliminaci�n de ruido al hacer zoom! Comprobar con la original

  //Filtro de baja frecuencia (paso bajo) = desenfoque, interpolaci�n, eliminaci�n de ruido
  /*(*h_filter)[0] = 1. / 25; (*h_filter)[1] = 1. / 25; (*h_filter)[2] = 1. / 25; (*h_filter)[3] = 1. / 25; (*h_filter)[4] = 1. / 25;
  (*h_filter)[5] = 1. / 25; (*h_filter)[6] = 1. / 25; (*h_filter)[7] = 1. / 25; (*h_filter)[8] = 1. / 25; (*h_filter)[9] = 1. / 25;
  (*h_filter)[10] = 1. / 25; (*h_filter)[11] = 1. / 25; (*h_filter)[12] = 1. / 25; (*h_filter)[13] = 1. / 25; (*h_filter)[14] = 1. / 25;
  (*h_filter)[15] = 1. / 25; (*h_filter)[16] = 1. / 25; (*h_filter)[17] = 1. / 25; (*h_filter)[18] = 1. / 25; (*h_filter)[19] = 1. / 25;
  (*h_filter)[20] = 1. / 25; (*h_filter)[21] = 1. / 25; (*h_filter)[22] = 1. / 25; (*h_filter)[23] = 1. / 25; (*h_filter)[24] = 1. / 25;*/
  
  //Interpolaci�n ponderada
  /*(*h_filter)[0] = 1. / 36; (*h_filter)[1] = 1. / 36; (*h_filter)[2] = 1. / 36; (*h_filter)[3] = 1. / 36; (*h_filter)[4] = 1. / 36;
  (*h_filter)[5] = 1. / 36; (*h_filter)[6] = 2. / 36; (*h_filter)[7] = 2. / 36; (*h_filter)[8] = 2. / 36;  (*h_filter)[9] = 1. / 36;
  (*h_filter)[10] = 1. / 36; (*h_filter)[11] = 2. / 36; (*h_filter)[12] = 4. / 36; (*h_filter)[13] = 2. / 36; (*h_filter)[14] = 1. / 36;
  (*h_filter)[15] = 1. / 36; (*h_filter)[16] = 2. / 36; (*h_filter)[17] = 2. / 36; (*h_filter)[18] = 2. / 36; (*h_filter)[19] = 1. / 36;
  (*h_filter)[20] = 1. / 36; (*h_filter)[21] = 1. / 36; (*h_filter)[22] = 1. / 36; (*h_filter)[23] = 1. / 36; (*h_filter)[24] = 1. / 36;*/

  //ESTE M�TODO LO HE SACADO DE SU P�GINA Y DEBER�A SUAVIZAR UN POCO LA IMAGEN

  //Filtro paso bajo = suavizado
  /*(*h_filter)[0] = 1.; (*h_filter)[1] = 1.; (*h_filter)[2] = 1.; (*h_filter)[3] = 1.; (*h_filter)[4] = 1.;
  (*h_filter)[5] = 1.; (*h_filter)[6] = 4.; (*h_filter)[7] = 4.; (*h_filter)[8] = 4.;  (*h_filter)[9] = 1.;
  (*h_filter)[10] = 1.; (*h_filter)[11] = 4.; (*h_filter)[12] = 12.; (*h_filter)[13] = 4.; (*h_filter)[14] = 1.;
  (*h_filter)[15] = 1.; (*h_filter)[16] = 4.; (*h_filter)[17] = 4.; (*h_filter)[18] = 4.; (*h_filter)[19] = 1.;
  (*h_filter)[20] = 1.; (*h_filter)[21] = 1.; (*h_filter)[22] = 1.; (*h_filter)[23] = 1.; (*h_filter)[24] = 1.;*/

  //ESTE TAMBI�N LO HE CONSEGUIDO DE UNA FUENTE EXTERNA Y EL RESULTADO ES ALOCADO COMO EN EL CASO ANTERIOR

  //Filtro Gaussiano -> Quedaba mejor con el c�digo mal. El resultado actual no me convence
  (*h_filter)[0] = 1.; (*h_filter)[1] = 4.; (*h_filter)[2] = 7.; (*h_filter)[3] = 4.; (*h_filter)[4] = 1.;
  (*h_filter)[5] = 4.; (*h_filter)[6] = 20.; (*h_filter)[7] = 33.; (*h_filter)[8] = 20.; (*h_filter)[9] = 4.;
  (*h_filter)[10] = 7.; (*h_filter)[11] = 33.; (*h_filter)[12] = 55.; (*h_filter)[13] = 33.; (*h_filter)[14] = 7.;
  (*h_filter)[15] = 4.; (*h_filter)[16] = 20.; (*h_filter)[17] = 33.; (*h_filter)[18] = 20.; (*h_filter)[19] = 4.;
  (*h_filter)[20] = 1.; (*h_filter)[21] = 4.; (*h_filter)[22] = 7.; (*h_filter)[23] = 4.; (*h_filter)[24] = 1.;

  /*(*h_filter)[0] = 2.; (*h_filter)[1] = 4.; (*h_filter)[2] = 7.; (*h_filter)[3] = 4.; (*h_filter)[4] = 1.;
  (*h_filter)[5] = 4.; (*h_filter)[6] = 9.; (*h_filter)[7] = 12.; (*h_filter)[8] = 9.; (*h_filter)[9] = 4.;
  (*h_filter)[10] = 5.; (*h_filter)[11] = 12.; (*h_filter)[12] = 15.; (*h_filter)[13] = 12.; (*h_filter)[14] = 5.;
  (*h_filter)[15] = 4.; (*h_filter)[16] = 9.; (*h_filter)[17] = 12.; (*h_filter)[18] = 9.; (*h_filter)[19] = 4.;
  (*h_filter)[20] = 2.; (*h_filter)[21] = 4.; (*h_filter)[22] = 5.; (*h_filter)[23] = 4.; (*h_filter)[24] = 1.;*/


  //ESTE LO HE SACADO DE SU P�GINA Y SE VE TODO NEGRO

  //Filtro de nitidez = No me gusta el resultado.
  /*(*h_filter)[0] = 0; (*h_filter)[1] = -1.; (*h_filter)[2] = -1.; (*h_filter)[3] = -1.; (*h_filter)[4] = 0;
  (*h_filter)[5] = -1.; (*h_filter)[6] = 2.; (*h_filter)[7] = -4.; (*h_filter)[8] = 2.;  (*h_filter)[9] = -1.;
  (*h_filter)[10] = -1.; (*h_filter)[11] = -4.; (*h_filter)[12] = 13.; (*h_filter)[13] = -4.; (*h_filter)[14] = -1.;
  (*h_filter)[15] = -1.; (*h_filter)[16] = 2.; (*h_filter)[17] = -4.; (*h_filter)[18] = 2.; (*h_filter)[19] = -1.;
  (*h_filter)[20] = 0; (*h_filter)[21] = -1.; (*h_filter)[22] = -1.; (*h_filter)[23] = -1.; (*h_filter)[24] = 0;*/


  //Detecci�n de bordes
  /*(*h_filter)[0] = 0.; (*h_filter)[1] = 0.; (*h_filter)[2] = 0.; (*h_filter)[3] = 0.; (*h_filter)[4] = 0.;
  (*h_filter)[5] = 0.; (*h_filter)[6] = 0.; (*h_filter)[7] = 1.; (*h_filter)[8] = 0.;  (*h_filter)[9] = 0.;
  (*h_filter)[10] = 0.; (*h_filter)[11] = 1.; (*h_filter)[12] = -4.; (*h_filter)[13] = 1.; (*h_filter)[14] = 0.;
  (*h_filter)[15] = 0.; (*h_filter)[16] = 0.; (*h_filter)[17] = 1.; (*h_filter)[18] = 0.; (*h_filter)[19] = 0.;
  (*h_filter)[20] = 0.; (*h_filter)[21] = 0.; (*h_filter)[22] = 0.; (*h_filter)[23] = 0.; (*h_filter)[24] = 0.;*/

  /*(*h_filter)[0] = 0.; (*h_filter)[1] = 0.; (*h_filter)[2] = 0.; (*h_filter)[3] = 0.; (*h_filter)[4] = 0.;
  (*h_filter)[5] = 0.; (*h_filter)[6] = -1.; (*h_filter)[7] = -1.; (*h_filter)[8] = -1.;  (*h_filter)[9] = 0.;
  (*h_filter)[10] = 0.; (*h_filter)[11] = -1.; (*h_filter)[12] = 8.; (*h_filter)[13] = -1.; (*h_filter)[14] = 0.;
  (*h_filter)[15] = 0.; (*h_filter)[16] = -1.; (*h_filter)[17] = -1.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0.;
  (*h_filter)[20] = 0.; (*h_filter)[21] = 0.; (*h_filter)[22] = 0.; (*h_filter)[23] = 0.; (*h_filter)[24] = 0.;*/

  /*(*h_filter)[0] = 0.; (*h_filter)[1] = 0.; (*h_filter)[2] = 0.; (*h_filter)[3] = 0.; (*h_filter)[4] = 0.;
  (*h_filter)[5] = 0.; (*h_filter)[6] = -1.; (*h_filter)[7] = 0.; (*h_filter)[8] = 1.;  (*h_filter)[9] = 0.;
  (*h_filter)[10] = 0.; (*h_filter)[11] = -1.; (*h_filter)[12] = 0.; (*h_filter)[13] = 1.; (*h_filter)[14] = 0.;
  (*h_filter)[15] = 0.; (*h_filter)[16] = -1.; (*h_filter)[17] = 0.; (*h_filter)[18] = 1.; (*h_filter)[19] = 0.;
  (*h_filter)[20] = 0.; (*h_filter)[21] = 0.; (*h_filter)[22] = 0.; (*h_filter)[23] = 0.; (*h_filter)[24] = 0.;*/
  //TODO: crear los filtros segun necesidad
  //NOTA: cuidado al establecer el tama�o del filtro a utilizar

}


void convolution(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redFiltered, 
                        unsigned char *d_greenFiltered, 
                        unsigned char *d_blueFiltered,
                        const int filterWidth)
{
  //TODO: Calcular tama�os de bloque
  const dim3 blockSize(16,16,1);
  const dim3 gridSize((numCols / blockSize.x) + 1, (numRows / blockSize.y) + 1, 1);

  //TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
  separateChannels << < gridSize, blockSize >> >(d_inputImageRGBA, numRows, numCols, d_redFiltered, d_greenFiltered, d_blueFiltered);

  //TODO: Ejecutar convoluci�n. Una por canal
  box_filter << <gridSize, blockSize >> > (d_redFiltered, d_red, numRows, numCols, filterWidth);
  box_filter << <gridSize, blockSize >> > (d_greenFiltered, d_green, numRows, numCols, filterWidth);
  box_filter << <gridSize, blockSize >> > (d_blueFiltered, d_blue, numRows, numCols, filterWidth);

  // Recombining the results. 
  //recombineChannels << <gridSize, blockSize >> >(d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA, numRows, numCols);
  recombineChannels << <gridSize, blockSize >> >(d_red, d_green, d_blue, d_outputImageRGBA, numRows, numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //system("pause");
}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}

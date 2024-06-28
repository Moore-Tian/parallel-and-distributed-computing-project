#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <cassert>

#include "kmeans.h"

using namespace std;

const int MAX_CHAR_PER_LINE = 1024;

class KMEANS
{
private:
	int numClusters;
	int numCoords;
	int numObjs;
	int *membership;
	char *filename; 
	float **objects;
	float **clusters;
	float threshold;
	int loop_iterations;

public:
	KMEANS(int k);
	void file_read(char *fn);
	void file_write();
	void cuda_kmeans();
	inline int nextPowerOfTwo(int n);
	void free_memory();
	virtual ~KMEANS();
};

KMEANS::~KMEANS()
{
	free(membership);
	free(clusters[0]);
	free(clusters);
	free(objects[0]);
	free(objects);
}

KMEANS::KMEANS(int k)
{
	threshold = 0.001;
	numObjs = 0;
	numCoords = 0;
	numClusters = k;
	filename = NULL;
	loop_iterations = 0;
}

void KMEANS::file_write()
{
	FILE *fptr;
	char outFileName[1024];

	sprintf(outFileName,"%s.cluster_centres",filename);
	printf("Writingcoordinates of K=%d cluster centers to file \"%s\"\n",numClusters,outFileName);
	fptr = fopen(outFileName,"w");
	for(int i=0;i<numClusters;i++)
	{
		fprintf(fptr,"%d ",i)	;
		for(int j=0;j<numCoords;j++)
			fprintf(fptr,"%f ",clusters[i][j]);
		fprintf(fptr,"\n");
	}
	fclose(fptr);

	sprintf(outFileName,"%s.membership",filename);
	printf("writing membership of N=%d data objects to file \"%s\" \n",numObjs,outFileName);
	fptr = fopen(outFileName,"w");
	for(int i=0;i<numObjs;i++)
	{
		fprintf(fptr,"%d %d\n",i,membership[i])	;
	}
	fclose(fptr);
}

inline int KMEANS::nextPowerOfTwo(int n)
{
	n--;
	n = n >> 1 | n;
	n = n >> 2 | n;
	n = n >> 4 | n;
	n = n >> 8 | n;
	n = n >> 16 | n;
	return ++n;
}

__host__ __device__ inline static 
float euclid_dist_2(int numCoords,int numObjs,int numClusters,float *objects,float *clusters,int objectId,int clusterId)
{
	int i;
	float ans = 0;
	for( i=0;i<numCoords;i++ )
	{
		ans += ( objects[numObjs * i + objectId] - clusters[numClusters*i + clusterId] ) *
			   ( objects[numObjs * i + objectId] - clusters[numClusters*i + clusterId] ) ;
	}
	return ans;
}

__global__ static void compute_delta(int *deviceIntermediates,int numIntermediates,	int numIntermediates2)
{
	extern __shared__ unsigned int intermediates[];

	intermediates[threadIdx.x] = (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0 ;
	__syncthreads();
	for(unsigned int s = numIntermediates2 /2 ; s > 0 ; s>>=1)
	{
		if(threadIdx.x < s)	
		{
			intermediates[threadIdx.x] += intermediates[threadIdx.x + s];	
		}
		__syncthreads();
	}
	if(threadIdx.x == 0)
	{
		deviceIntermediates[0] = intermediates[0];
	}
}

__global__ static void find_nearest_cluster(int numCoords,int numObjs,int numClusters,float *objects, float *deviceClusters,int *membership ,int *intermediates)
{
	extern __shared__ char sharedMemory[];
	unsigned char *membershipChanged = (unsigned char *)sharedMemory;
	float *clusters = deviceClusters;

	membershipChanged[threadIdx.x] = 0;

	int objectId = blockDim.x * blockIdx.x + threadIdx.x;
	if( objectId < numObjs )
	{
		int index;
		float dist,min_dist;
		index = 0;
		min_dist = euclid_dist_2(numCoords,numObjs,numClusters,objects,clusters,objectId,0);
		
		for(int i=0;i<numClusters;i++)
		{
			dist = euclid_dist_2(numCoords,numObjs,numClusters,objects,clusters,objectId,i)	;
			if( dist < min_dist )
			{
				min_dist = dist;
				index = i;
			}
		}

		if( membership[objectId]!=index )
		{
			membershipChanged[threadIdx.x] = 1;	
		}
		membership[objectId] = index;

		__syncthreads();

#if 1
		for(unsigned int s = blockDim.x / 2; s > 0 ;s>>=1)
		{
			if(threadIdx.x < s)	
			{
				membershipChanged[threadIdx.x] += membershipChanged[threadIdx.x + s];//calculate all changed values and save result to membershipChanged[0]
			}
			__syncthreads();
		}
		if(threadIdx.x == 0)
		{
			intermediates[blockIdx.x] = membershipChanged[0];
		}
#endif
	}
}

void KMEANS::cuda_kmeans()
{
	int index,loop = 0;
	int *newClusterSize;
	float delta;
	float **dimObjects;
	float **dimClusters;
	float **newClusters;

	float *deviceObjects;
	float *deviceClusters;
	int *deviceMembership;
	int *deviceIntermediates;

	malloc2D(dimObjects,numCoords,numObjs,float);
	for(int i=0;i<numCoords;i++)
	{
		for(int j=0;j<numObjs;j++)
		{
			dimObjects[i][j] = objects[j][i];	
		}
	}

	malloc2D(dimClusters, numCoords, numClusters,float);
	for(int i=0;i<numCoords;i++)
	{
		for(int j=0;j<numClusters;j++)
		{
			dimClusters[i][j] = dimObjects[i][j];
		}
	}
	newClusterSize = new int[numClusters];
	assert(newClusterSize!=NULL);
	malloc2D(newClusters,numCoords,numClusters,float);
	memset(newClusters[0],0,numCoords * numClusters * sizeof(float) );
	
	const unsigned int numThreadsPerClusterBlock = 32;
	const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock -1)/numThreadsPerClusterBlock;
	const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);

	const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(unsigned char);

	const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int);

	cudaMalloc(&deviceObjects,numObjs*numCoords*sizeof(float));
	cudaMalloc(&deviceClusters,numClusters*numCoords*sizeof(float));
	cudaMalloc(&deviceMembership,numObjs*sizeof(int));
	cudaMalloc(&deviceIntermediates,numReductionThreads*sizeof(unsigned int));

	cudaMemcpy(deviceObjects,dimObjects[0],numObjs*numCoords*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMembership,membership,numObjs*sizeof(int),cudaMemcpyHostToDevice);

	do
	{
		cudaMemcpy(deviceClusters,dimClusters[0],numClusters*numCoords*sizeof(float),cudaMemcpyHostToDevice);

		find_nearest_cluster<<<numClusterBlocks,numThreadsPerClusterBlock,clusterBlockSharedDataSize>>>(numCoords,numObjs,numClusters,deviceObjects,deviceClusters,deviceMembership,deviceIntermediates);

		cudaDeviceSynchronize();

		compute_delta<<<1,numReductionThreads,reductionBlockSharedDataSize>>>(deviceIntermediates,numClusterBlocks,numReductionThreads);

		cudaDeviceSynchronize();
		
		int d;
		cudaMemcpy(&d,deviceIntermediates,sizeof(int),cudaMemcpyDeviceToHost);
		delta = (float)d;

		cudaMemcpy(membership,deviceMembership,numObjs*sizeof(int),cudaMemcpyDeviceToHost);
		
		for(int i=0;i<numObjs;i++)
		{
			index = membership[i];
			newClusterSize[index]++;
			for(int j=0;j<numCoords;j++)
			{
				newClusters[j][index] += objects[i][j];
			}
		}
		for(int i=0;i<numClusters;i++)
		{
			for(int j=0;j<numCoords;j++)
			{
				if(newClusterSize[i] > 0)	
					dimClusters[j][i] = newClusters[j][i]/newClusterSize[i];
				newClusters[j][i] = 0.0;
			}
			newClusterSize[i] = 0 ;
		}
		delta /= numObjs;
	}while( delta > threshold && loop++ < 500 );

	loop_iterations = loop + 1;
	
	malloc2D(clusters,numClusters,numCoords,float);
	for(int i=0;i<numClusters;i++)
	{
		for(int j=0;j<numCoords;j++)
		{
			clusters[i][j] = dimClusters[j][i];
		}
	}

	cudaFree(deviceObjects)	;
	cudaFree(deviceClusters);
	cudaFree(deviceMembership);
	cudaFree(deviceMembership);

	free(dimObjects[0]);
	free(dimObjects);
	free(dimClusters[0]);
	free(dimClusters);
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
}

void KMEANS::file_read(char *fn)
{

	FILE *infile;
	char *line = new char[MAX_CHAR_PER_LINE];
	int lineLen = MAX_CHAR_PER_LINE;

	filename = fn;
	infile = fopen(filename,"r");
	assert(infile!=NULL);
	while( fgets(line,lineLen,infile) )
	{
		numObjs++;	
	}
	rewind(infile);
	while( fgets(line,lineLen,infile)!=NULL )
	{
		if( strtok(line," \t\n")!=0 )	
		{
			while( strtok(NULL," \t\n") )	
				numCoords++;
			break;
		}
	}
	rewind(infile);
	objects = new float*[numObjs];
	for(int i=0;i<numObjs;i++)
	{
		objects[i] = new float[numCoords];
	}
	int i=0;
	while( fgets(line,lineLen,infile)!=NULL )
	{
		if( strtok(line," \t\n") ==NULL ) continue;
		for(int j=0;j<numCoords;j++)
		{
			objects[i][j] = atof( strtok(NULL," ,\t\n") )	;
		}
		i++;
	}
	
	membership = new int[numObjs];
	assert(membership!=NULL);
	for(int i=0;i<numObjs;i++)
		membership[i] = -1;
}

int main(int argc,char *argv[])
{
	KMEANS kmeans(atoi(argv[1]));
	kmeans.file_read(argv[2]);
	kmeans.cuda_kmeans();
	kmeans.file_write();
	return 0;
}

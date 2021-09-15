#include <string>
#include <vector>

#define DIM 200			  //number of couples string and seq
#define maxseq 204800	  //dimension of seq array
#define maxstring 1638400 //dimension of string array
#define PI maxseq + DIM   //dimension of PI array

#include <iostream>
extern "C"
{
	void kmp(int *occ, int *stringdim, int *seqdim, char *seq, char *string, int *pi)
	{
		//Set HLS INTERFACES
#pragma HLS INTERFACE m_axi port = occ offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = stringdim offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = seqdim offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = seq offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = string offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = pi offset = slave bundle = gmem5

#pragma HLS INTERFACE s_axilite port = occ bundle = control
#pragma HLS INTERFACE s_axilite port = stringdim bundle = control
#pragma HLS INTERFACE s_axilite port = seqdim bundle = control
#pragma HLS INTERFACE s_axilite port = seq bundle = control
#pragma HLS INTERFACE s_axilite port = string bundle = control
#pragma HLS INTERFACE s_axilite port = pi bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS dataflow

		//Initiliase arrays
		//Occurences
		int occ_local[DIM];
		//String array containing the input strings
		char string_local[maxstring];
		//Seq array cointaining the input sequences
		char seq_local[maxseq];
		//String dim containing the dimension of the input strings of the array
		int stringdim_local[DIM];
		//Seq dim containing the dimension of the input sequences of the array
		int seqdim_local[DIM];
		//Pi contanins the failure table computed for all the sequences in seq
		int pi_local[PI];

		std::cout << "SOURCE KMP" << std::endl;

		//Copy the content of the Buffers to local arrays

		for (int i = 0; i < PI; i++)
		{
#pragma HLS pipeline
			pi_local[i] = pi[i];
		}

		for (int i = 0; i < DIM; i++)
		{
#pragma HLS pipeline
			occ_local[i] = occ[i];
			stringdim_local[i] = stringdim[i];
			seqdim_local[i] = seqdim[i];
		}

		for (int i = 0; i < maxstring; i++)
		{
#pragma HLS pipeline
			string_local[i] = string[i];
		}

		for (int i = 0; i < maxseq; i++)
		{
#pragma HLS pipeline
			seq_local[i] = seq[i];
		}


		//Define utility variables to execute the matching and slide between sequences
		unsigned int i = 0;
		unsigned int j = 0;
		unsigned int a = 0;
		unsigned int b = 0;
		int pi_count = 0;
		int n;
		int k;

//For all couples string and sequence
//The outer loop is flattening with the inner
	kmp:
		for (n = 0; n < DIM; n++)
		{

//Perform the matching
		matching:
			for (k = 0; k < (maxstring + maxseq); k++)
			{
//HLS pipeline to execute multiple indipendent instructions in parallel
#pragma HLS pipeline

				if (j <= seqdim_local[n])
				{

					if (i >= (stringdim_local[n]))
					{

						a = a + seqdim_local[n];
						b = b + stringdim_local[n];
						pi_count = pi_count + 1 + seqdim_local[n];
						i = 0;
						j = 0;

						break;
					}

					if (string_local[b + i + j] == seq_local[a + j])
					{
						j++;

						if (j == seqdim_local[n])
						{
							//If a matching occures, update the array of occurentces with the index of the string
							occ_local[n] = i;
							a = a + seqdim_local[n];
							b = b + stringdim_local[n];
							pi_count = pi_count + 1 + seqdim_local[n];
							i = 0;
							j = 0;

							break;
						}
					}
					else
					{

						if (i >= (stringdim_local[n]))
						{

							a = a + seqdim_local[n];
							b = b + stringdim_local[n];
							pi_count = pi_count + 1 + seqdim_local[n];
							i = 0;
							j = 0;

							break;
						}
						else
						{

							if (j == 0)
								i++;
							else
							{

								i = i + j - pi_local[j + pi_count];
								j = 0;
							}
						}
					}
				}
				else
				{

					a = a + seqdim_local[n];
					b = b + stringdim_local[n];
					pi_count = pi_count + 1 + seqdim_local[n];
					i = 0;
					j = 0;

					break;
				}
			}
		}

		//Copy the result of all the matches to the buffer and return it

		for (int i = 0; i < DIM; i++)
		{
#pragma HLS pipeline
			occ[i] = occ_local[i];
		}

		/*
		std::cout << "OCC FINAL" << std::endl;
		for (int i = 0; i < DIM; i++)
		{
			std::cout << i << " " << occ_local[i] << std::endl;
		}

*/
	}
}

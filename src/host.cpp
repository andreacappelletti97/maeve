#include "xcl2.hpp"
#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#define DIM 200           //number of couples string and seq
#define maxseq 204800     //dimension of seq array
#define maxstring 1638400 //dimension of string array
#define PI maxseq + DIM   //dimension of PI array

//Test function
bool check(std::string v1, std::string v2, int occ)
{

  std::vector<int> test_occ(v1.size(), 0);

  unsigned int idx = 0;

  test_occ[0] = -1;

  for (unsigned int i = 0; i < v1.size(); i++)
  {
    bool tmp = true;
    for (unsigned int j = 0; j < v2.size(); j++)
    {
      if (v1[i + j] != v2[j] || (i + j) > v1.size())
      {
        tmp = false;
        break;
      }
    }
    if (tmp)
    {
      test_occ[idx] = i;
      idx++;
    }
  }

  bool test = true;

  if (test_occ[0] != occ)
    test = false;

  // std::cout<<test_occ[0]<<std::endl;

  // if (test_occ[0]==-1) //AGGIUNTA
  // 	std::cout<<"NO MATCH"<<std::endl;

  return test;
}

double run_krnl(cl::Context &context,
                cl::CommandQueue &q,
                cl::Kernel &kernel,
                std::vector<int, aligned_allocator<int>> &vStringdim,
                std::vector<int, aligned_allocator<int>> &vSeqdim,
                std::vector<char, aligned_allocator<char>> &vString,
                std::vector<char, aligned_allocator<char>> &vSeq,
                std::vector<int, aligned_allocator<int>> &vPi,
                std::vector<int, aligned_allocator<int>> &vOcc)
{
  cl_int err;

  // These commands will allocate memory on the FPGA. The cl::Buffer objects can
  // be used to reference the memory locations on the device.
  //Creating Buffers

  std::cout << "RUN KERNEL ENTER" << std::endl;

  OCL_CHECK(err,
            cl::Buffer buffer_stringdim(context,
                                        CL_MEM_USE_HOST_PTR |
                                            CL_MEM_READ_ONLY,
                                        sizeof(int) * DIM,
                                        vStringdim.data(),
                                        &err));
  OCL_CHECK(err,
            cl::Buffer buffer_seqdim(context,
                                     CL_MEM_USE_HOST_PTR |
                                         CL_MEM_READ_ONLY,
                                     sizeof(int) * DIM,
                                     vSeqdim.data(),
                                     &err));

  OCL_CHECK(err,
            cl::Buffer buffer_string(context,
                                     CL_MEM_USE_HOST_PTR |
                                         CL_MEM_READ_ONLY,
                                     sizeof(char) * maxstring,
                                     vString.data(),
                                     &err));

  OCL_CHECK(err,
            cl::Buffer buffer_seq(context,
                                  CL_MEM_USE_HOST_PTR |
                                      CL_MEM_READ_ONLY,
                                  sizeof(char) * maxseq,
                                  vSeq.data(),
                                  &err));
  OCL_CHECK(err,
            cl::Buffer buffer_pi(context,
                                 CL_MEM_USE_HOST_PTR |
                                     CL_MEM_READ_ONLY,
                                 sizeof(int) * PI,
                                 vPi.data(),
                                 &err));

  OCL_CHECK(err,
            cl::Buffer buffer_occ(context,
                                  CL_MEM_USE_HOST_PTR |
                                      CL_MEM_WRITE_ONLY,
                                  sizeof(int) * DIM,
                                  vOcc.data(),
                                  &err));

  //Setting the kernel Arguments

  OCL_CHECK(err, err = (kernel).setArg(0, buffer_occ));
  OCL_CHECK(err, err = (kernel).setArg(1, buffer_stringdim));
  OCL_CHECK(err, err = (kernel).setArg(2, buffer_seqdim));
  OCL_CHECK(err, err = (kernel).setArg(3, buffer_seq));
  OCL_CHECK(err, err = (kernel).setArg(4, buffer_string));
  OCL_CHECK(err, err = (kernel).setArg(5, buffer_pi));

  std::cout << "ARG SETUP DONE" << std::endl;

  // Copy input data to Device Global Memory from HOST to board
  OCL_CHECK(err,
            err = q.enqueueMigrateMemObjects({buffer_occ, buffer_stringdim, buffer_seqdim, buffer_string, buffer_seq, buffer_pi},
                                             0 /* 0 means from host*/));

  std::cout << "INPUT DATA COPIED" << std::endl;

  std::chrono::duration<double> kernel_time(0);

  auto kernel_start = std::chrono::high_resolution_clock::now();

  //Execution of the kernel KMP
  OCL_CHECK(err, err = q.enqueueTask(kernel));

  auto kernel_end = std::chrono::high_resolution_clock::now();

  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

  std::cout << "KERNEL EXE COMPLETED" << std::endl;

  // Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err,
            err = q.enqueueMigrateMemObjects({buffer_occ},
                                             CL_MIGRATE_MEM_OBJECT_HOST));

  q.finish();

  return kernel_time.count();
}

//Function to compute the failure function of the KMP algorithm
void failure_function(std::vector<char, aligned_allocator<char>> &seq, std::vector<int, aligned_allocator<int>> &seqdim, std::vector<int, aligned_allocator<int>> &pi)
{

  int seq_count = 0; //scan the sequences
  int pi_count = 0;  //scan the failure table

  //For each couple computer PI
  for (int n = 0; n < 1; n++)
  {
    pi[pi_count] = -1;    //first element is always set to -1
    pi[pi_count + 1] = 0; //second element is always set to 0

    for (int i = 1; i < seqdim[n]; i++)
    {
      if (seq[seq_count + i] == seq[seq_count + pi[pi_count + i]])
        pi[pi_count + i + 1] = pi[pi_count + i] + 1;
      else
      {
        if (seq[seq_count + i] == seq[seq_count])
          pi[pi_count + i + 1] = pi[pi_count + i];
        else
          pi[pi_count + i + 1] = 0;
      }
    }

    pi_count = pi_count + seqdim[n] + 1;
    seq_count = seq_count + seqdim[n];
  }
}

//Utility Function to print vector content
void printVectorContent(std::vector<char, aligned_allocator<char>> &string)
{
  for (unsigned int i = 0; i < string.size(); i++)
  {
    std::cout << i << " " << string[i] << std::endl;
  }
}

//Function to read fasta file format and automatically set dimension of seq and string
void readFastaInput(std::vector<char, aligned_allocator<char>> &seq, std::vector<int, aligned_allocator<int>> &seqdim, bool isString, std::vector<std::string, aligned_allocator<std::string>> &shortSeq)
{
  //Input files are located into the directory ./input
  std::ifstream input;
  if (isString)
  {
    input.open("./input/string.fasta");
  }
  else
  {
    input.open("./input/pattern.fasta");
  }

  std::string line, id, DNA_sequence;

  while (std::getline(input, line))
  {
    //std::cout << line << std::endl;

    // line may be empty so you *must* ignore blank lines
    // or you have a crash waiting to happen with line[0]
    if (line.empty())
      continue;

    if (line[0] == '>')
    {
      // output previous line before overwriting id
      // but ONLY if id actually contains something
      if (!id.empty())
      {
        for (int i = 0; i < DNA_sequence.size(); i++)
        {
          seq.push_back(DNA_sequence[i]);
        }
        seqdim.push_back(DNA_sequence.size());
        //std::cout << "DNA SIZE: " << DNA_sequence.size() << std::endl;
      }

      //seqdim.push_back(DNA_sequence.size());
      id = line.substr(1);
      shortSeq.push_back(id);
      DNA_sequence.clear();
    }
    else
    { //  if (line[0] != '>'){ // not needed because implicit
      DNA_sequence += line;
    }
  }
  for (int i = 0; i < DNA_sequence.size(); i++)
  {
    seq.push_back(DNA_sequence[i]);
  }
  seqdim.push_back(DNA_sequence.size());
  /*
  std::cout << "PRINT SEQ " << std::endl;
  for (int i = 0; i < seq.size(); i++)
  {
    std::cout << seq[i];
  } 

  std::cout << std::endl;
  std::cout << "PRINT SEQDIM " << std::endl;
  for (int i = 0; i < seqdim.size(); i++)
  {
    std::cout << i << " " << seqdim[i] << std::endl;
  }
  */
}

int main(int argc, char **argv)
{
  //Define the platform = devices + context + queues
  cl_int err;
  cl::Context context;
  cl::CommandQueue q;
  cl::Kernel kernel;
  std::string binaryFile = argv[1];

  // The get_xil_devices will return vector of Xilinx Devices
  auto devices = xcl::get_devices("Xilinx");

  // read_binary_file() command will find the OpenCL binary file created using the
  // V++ compiler load into OpenCL Binary and return pointer to file buffer.

  auto fileBuf = xcl::read_binary_file(binaryFile);

  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  int valid_device = 0;
  for (unsigned int i = 0; i < devices.size(); i++)
  {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err,
              q = cl::CommandQueue(
                  context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS)
    {
      std::cout << "Failed to program device[" << i
                << "] with xclbin file!\n";
    }
    else
    {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err,
                kernel = cl::Kernel(program, "kmp", &err));
      valid_device++;
      break; // we break because we found a valid device
    }
  }
  if (valid_device == 0)
  {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  //Declare input vector
  std::vector<int, aligned_allocator<int>> occ(DIM);
  std::vector<int, aligned_allocator<int>> stringdim;
  std::vector<int, aligned_allocator<int>> seqdim;
  std::vector<char, aligned_allocator<char>> seq;
  std::vector<char, aligned_allocator<char>> string;
  std::vector<int, aligned_allocator<int>> pi;

  //Declare a vector to store the IDs of the input string
  std::vector<std::string, aligned_allocator<std::string>> inputId;
  //Declare a vector to store the IDs of the sequences
  std::vector<std::string, aligned_allocator<std::string>> shortSeq;

  //Read the FASTA files in input
  readFastaInput(string, stringdim, true, inputId);
  readFastaInput(seq, seqdim, false, shortSeq);
  //Delete the last two elements of shortSeq
  shortSeq.pop_back();
  shortSeq.pop_back();

  //Initialise vector of boolean to keep track of the occurences found
  std::vector<bool> seqFound(seqdim.size());

  for (size_t i = 0; i < seqdim.size(); i++)
  {
    seqFound[i] = false;
  }
  //Take the execution time of the kerlen with preprocessing of the input
  double kernel_time_in_sec = 0;

  //Starting from here we populate the string buffer and the sequence buffer taking into account
  //our kernel constraints in terms of dimension
  //More in detail we set
  //String dimension: 1638400
  //Seq dimension: 204800
  //Scanning all the arrays and populating it once execution at a time
  while (string.size() > 0)
  {
    //If the lenth of the string is greater or equal than 1638400 we populate the buffer with the maxstring size
    if (string.size() >= maxstring)
    {
      //Utility vectors to populate the buffers
      std::vector<char, aligned_allocator<char>> stringDiv(maxstring);
      std::vector<int, aligned_allocator<int>> stringdimDiv(DIM);

      //Copy all the content of string into stringDiv until maxstring is reached
      for (size_t i = 0; i < maxstring; i++)
      {
        stringDiv[i] = string[i];
      }
      //Set all the string dimension
      for (size_t m = 0; m < DIM; m++)
      {
        stringdimDiv[m] = maxstring / DIM;
      }

      int sequence_index_seq = 0;
      int save_sequence_index_seq = 0;
      //For all the sequences populate the array of sequences and scan them once at a time with the current strings
      for (size_t l = 0; l < seqdim.size(); l++)
      {
        //Reset the occurences found up to now
        for (int i = 0; i < DIM; i++)
        {
          occ[i] = -1;
        }
        //Utility arrays to run different sequences implementation
        std::vector<char, aligned_allocator<char>> currentSeq(seqdim[l] * DIM);
        std::vector<int, aligned_allocator<int>> currentSeqDim(DIM);
        std::vector<int, aligned_allocator<int>> pi(seqdim[l] * DIM + DIM);
        int sequence_index = 0;

        save_sequence_index_seq = sequence_index_seq;
        //Fill the DIM couples for the comparisons
        for (size_t i = 0; i < DIM; i++)
        {
          for (size_t s = sequence_index; s < sequence_index + seqdim[l]; s++)
          {

            currentSeq[s] = seq[sequence_index_seq++];
          }
          sequence_index += seqdim[l];
          currentSeqDim[i] = seqdim[l];
          sequence_index_seq = save_sequence_index_seq;
        }

        sequence_index_seq = save_sequence_index_seq + seqdim[l];
        //Call the failure function for the current sequence and compute it
        failure_function(currentSeq, currentSeqDim, pi);
        //Call the kernel for the KMP algorithm processing
        kernel_time_in_sec += run_krnl(context,
                                       q,
                                       kernel,
                                       stringdimDiv,
                                       currentSeqDim,
                                       stringDiv,
                                       currentSeq,
                                       pi,
                                       occ);
        //From here we are going to call the test function in order to understand if the result computed are correct
        std::string str;
        std::string sq;
        int a = 0;
        int b = 0;
        bool test = true;
        for (unsigned int i = 0; i < DIM; i++)
        {

          for (int j = 0; j < stringdimDiv[i]; j++)
            str.push_back(stringDiv[b + j]);
          for (int j = 0; j < currentSeqDim[i]; j++)
            sq.push_back(currentSeq[a + j]);

          std::string v1 = std::string(str);
          std::string v2 = std::string(sq);

          test &= check(v1, v2, occ[i]);

          str = "";
          sq = "";

          a = a + currentSeqDim[i];
          b = b + stringdimDiv[i];
        }

        if (test)
          std::cout << "ALL RESULTS CORRECT" << std::endl;
        else
          std::cout << "TEST FAILED" << std::endl;

        //For all the occurences found, we save the result
        for (size_t p = 0; p < DIM; p++)
        {
          if (occ[p] != -1)
          {
            seqFound[l] = true;
          }
        }
      }

      //Delete the part of the string already scanned up to now
      string.erase(string.begin(), string.begin() + maxstring);
    }
    //If the actual length of the string is less than maxstring this is the last loop iteration
    else
    {
      //Utility vectors to populate the buffers
      std::vector<char, aligned_allocator<char>> stringDiv(maxstring);
      std::vector<int, aligned_allocator<int>> stringdimDiv(DIM);
      //Populate the string vector until the current size is reached
      std::string str;
      for (size_t i = 0; i < string.size(); i++)
      {
        stringDiv[i] = string[i];
        str.push_back(string[i]);
      }
      stringdimDiv[0] = string.size();

      int seq_index = 0;
      //Scan all the sequences
      for (size_t l = 0; l < seqdim.size(); l++)
      {
        //Reset the occurences found up to now
        for (int i = 0; i < DIM; i++)
        {
          occ[i] = -1;
        }
        //Utility arrays to run different sequences implementation
        std::vector<char, aligned_allocator<char>> currentSeq(maxseq);
        std::vector<int, aligned_allocator<int>> currentSeqDim(DIM);
        std::vector<int, aligned_allocator<int>> pi(PI);

        for (size_t i = 0; i < seqdim[l]; i++)
        {
          currentSeq[i] = seq[seq_index++];
        }

        currentSeqDim[0] = seqdim[l];

        /*
        for (int i = 0; i < 3; i++)
        {
          std::cout << i << " " << stringDiv[i] << std::endl;
        }
        */

        //Call the failure function for the current sequence and compute it
        failure_function(currentSeq, currentSeqDim, pi);
        //Call the kernel for the KMP algorithm processing and sum up the total execution time
        kernel_time_in_sec += run_krnl(context,
                                       q,
                                       kernel,
                                       stringdimDiv,
                                       currentSeqDim,
                                       stringDiv,
                                       currentSeq,
                                       pi,
                                       occ);

        //From here we are going to call the test function in order to understand if the result computed are correct

        std::string sq;
        bool test = true;

        for (int j = 0; j < seqdim[l]; j++)
          sq.push_back(currentSeq[j]);

        /*
        std::cout << std::endl;

        std::cout << "str" << std::endl;
        std::cout << str << std::endl;
        std::cout << "sq" << std::endl;
        std::cout << sq << std::endl;
        std::cout << occ[0];

        std::cout << std::endl;

        */

        std::string v1 = std::string(str);
        std::string v2 = std::string(sq);

        test &= check(v1, v2, occ[0]);

        if (test)
          std::cout << "ALL RESULTS CORRECT" << std::endl;
        else
          std::cout << "TEST FAILED" << std::endl;

        //Delete the part of the string already scanned up to now

        string.erase(string.begin(), string.begin() + string.size());

        for (size_t p = 0; p < DIM; p++)
        {
          if (occ[p] != -1)
          {
            seqFound[l] = true;
          }
        }
      }
    }
    std::cout << string.size() << std::endl;
  }

  std::cout << "Total time in seconds: " << kernel_time_in_sec << std::endl;
  std::cout << "SEQUENCES FOUND" << std::endl;

  for (size_t i = 0; i < seqdim.size(); i++)
  {
    if (seqFound[i] == true)
    {
      std::cout << "Sequence: " << shortSeq[i] << std::endl;
    }
  }
}

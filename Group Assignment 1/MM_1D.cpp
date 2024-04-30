// Modified to work within the constraints of CS 4230

#include <stdio.h>
#include <iostream>
#include <string>
#include <ctype.h>
#include <chrono>
#include <fstream>
#include <iomanip>

#include <mpi.h>

using namespace std;

// Definitons
typedef chrono::duration<double> Runtime; // One of many possible ways to record the time

// Prototypes
int* MM_sequential(const int* A, const int* B, const int m, const int n, const int q);
int* MM_1D_Distributed(const int* A, const int* B, const int m, const int n, const int q);

void printMatrix(const int* A, const int m, const int n);
int* randomMatrix(const int m, const int n);
int* transposeMatrix(const int* A, const int m, const int n);
int compareMatrix( const int* A, const int *B, const int m, const int n);
string appendString(const string original, const int value);

// ------ SEND/RECV TAGS ------
int const ERROR_CHECK = 0;
int const SCATTER_1D_A = 1;
int const SCATTER_1D_B = 2;
int const PASSING_1D_B = 3;
int const GATHER_1D = 4;


int main(int argc, char*argv[])
{
	// ------ MPI SETUP ------
	int pid; // Processor ID
	int p_total; // Number of Processors Being Used

	// --------------- BEGIN MPI ---------------
	// MPI Start
	/*
	* DOESN'T CREATE THE THREADS
	* JUST ALLOWS FOR MPI FUNCTIONS TO EXIST AFTER
	* ALL THINGS PRIOR WILL BE DONE ON ALL PROCESSORS
	*/
	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &p_total);

	// User input for matrix
	int m = p_total;
	int n = p_total;
	int q = p_total;
	int deltaM = 2;
	int deltaN = 2;
	int deltaQ = 2;

	try
	{
		m = stoi(argv[1]);
		n = stoi(argv[1]);
		q = stoi(argv[1]);

		deltaM = stoi(argv[4]);
		deltaN = stoi(argv[5]);
		deltaQ = stoi(argv[6]);
	}
	catch (...) {}

	// If a dimension is not divisible by processors
	if ( m % p_total != 0 || n % p_total != 0 || q % p_total != 0)
	{
		cerr << "Bad Dimensions." << endl;
		MPI_Finalize();
		exit(1);
	}

	// If there are less dimensions than processors
	if ( m < p_total || n < p_total || q < p_total)
	{
		cerr << "Bad Processor Count." << endl;
		MPI_Finalize();
		exit(1);
	}

	// MASTER THREAD
	if ( pid == 0 )
	{
		bool ONERUN = true;

		try
		{
			if ( stoi(argv[7]) == 1)
			{
				ONERUN = true;
			}
		}
		catch (...) {}

		if( deltaM == 1 || deltaN == 1 || deltaQ == 1 )
		{
			ONERUN = true;
		}

		// Code for error broadcasting
		int broadcastCode = 0;

		// Unique filename for logging
		string filename = "1D";
		filename = appendString(filename, m);
		filename = appendString(filename, n);
		filename = appendString(filename, q);
		filename = filename + "_";
		filename = appendString(filename, deltaM);
		filename = appendString(filename, deltaN);
		filename = appendString(filename, deltaQ);
		filename = filename + ".txt";

		// Attempt to open the file
		ofstream outputfile(filename);

		// Failed to open file
		if (!outputfile.is_open())
		{
			cerr << "Error opening file." << endl;
			broadcastCode = 1;
			for (int i = 1; i < p_total; i++)
			{
				MPI_Send(&broadcastCode, 1, MPI_INT, i, ERROR_CHECK, MPI_COMM_WORLD);
			}

			MPI_Finalize();
			return 1;
		}
		else
		{
			for (int i = 1; i < p_total; i++)
			{
				MPI_Send(&broadcastCode, 1, MPI_INT, i, ERROR_CHECK, MPI_COMM_WORLD);
			}
		}

		// HEADERS
		outputfile << right
		           << setw(10) << "m"
		           << setw(10) << "n"
		           << setw(10) << "q"
		           << setw(5) << "p"
		           << setw(25) << "milliseconds"
		           << setw(25) << "HH:MM:SS.ms"
		           << endl;

		// ------- VARIABLE SETUP -------
		const int MAXIMUM_INTEGER = 30000;

		int* A_matrix;
		int* B_matrix;
		int* parallel_matrix;

		// Formatting timers
		auto start = chrono::high_resolution_clock::now();
		auto end = chrono::high_resolution_clock::now();

		do
		{
			// Two same sized matricies
			A_matrix = randomMatrix(m, n);
			B_matrix = randomMatrix(n, q);


			// ------ 1D ALGORITHM ------
			start = chrono::high_resolution_clock::now();
			parallel_matrix = MM_1D_Distributed(A_matrix, B_matrix, m, n, q);
			end = chrono::high_resolution_clock::now();

			// Calculate times
			chrono::duration<double, ratio<3600>> hours = (end - start);
			chrono::duration<double, ratio<60>> minutes = (end - start);
			chrono::duration<double, milli> milli = (end - start);
			chrono::duration<double> seconds = (end - start);

			// Logging
			outputfile << right
			           << setw(10) << m
			           << setw(10) << n
			           << setw(10) << q
			           << setw(5) << p_total
			           << setw(25) << milli.count() // For excel sheet later
			           << setw(15) << static_cast<int>(hours.count()) << ":" << setfill('0')
			           << setw(2) << static_cast<int>(minutes.count()) % 60 << ":"
			           << setw(2) << static_cast<int>(seconds.count()) % 60 << "."
			           << setw(3) << static_cast<int>(milli.count()) % 1000 << setfill(' ')
			           << endl;


			// Prep Next Iteration
			m *= deltaM;
			n *= deltaN;
			q *= deltaQ;

			// Terminal Checks
			// if( ONERUN )
			// {
			// 	printMatrix(A_matrix, m, n);
			// 	printMatrix(B_matrix, n, q);
			// 	printMatrix(parallel_matrix, m, q);
			// }
			
			if (m >= MAXIMUM_INTEGER || n >= MAXIMUM_INTEGER || q >= MAXIMUM_INTEGER || ONERUN)
			{
				broadcastCode = 1;
				for (int i = 1; i < p_total; i++)
				{
					MPI_Send(&broadcastCode, 1, MPI_INT, i, ERROR_CHECK, MPI_COMM_WORLD);
				}
			}
			else
			{
				for (int i = 1; i < p_total; i++)
				{
					MPI_Send(&broadcastCode, 1, MPI_INT, i, ERROR_CHECK, MPI_COMM_WORLD);
				}
			}

			free(A_matrix);
			free(B_matrix);
			free(parallel_matrix);
		}
		while (m < MAXIMUM_INTEGER && n < MAXIMUM_INTEGER && q < MAXIMUM_INTEGER && !ONERUN);

		outputfile.close();

	}
	else
	{
		// DO WHILE PROGRAM CONTINUES
		int number;
		MPI_Status status;

		// Look for error from open file
		MPI_Recv(&number, 1, MPI_INT, 0, ERROR_CHECK, MPI_COMM_WORLD, &status);

		if (number == 1)
		{
			MPI_Finalize();
			return 0;
		}


		do
		{
			// ------ 1D ALGORITHM ------
			// Get size of A
			int A_start = pid * m / p_total;
			int A_end = (pid + 1) * (m / p_total);
			int A_size = (A_end - A_start) * n;
			int* A_segment = (int*)malloc(A_size * sizeof(int));

			MPI_Recv(A_segment, A_size, MPI_INT, 0, SCATTER_1D_A, MPI_COMM_WORLD, &status);

			// Get size of B
			int B_start = pid * q / p_total;
			int B_end = (pid + 1) * (q / p_total);
			int B_size = (B_end - B_start) * n;
			int* B_segment = (int*)malloc(B_size * sizeof(int));

			MPI_Recv(B_segment, B_size, MPI_INT, 0, SCATTER_1D_B, MPI_COMM_WORLD, &status);

			// 1D Algorithm Call
			MM_1D_Distributed(A_segment, B_segment, m, n, q);

			// Finished 1D
			MPI_Recv(&number, 1, MPI_INT, 0, ERROR_CHECK, MPI_COMM_WORLD, &status);
			free(A_segment);
			free(B_segment);

			// Next Iteration
			m*= deltaM;
			n*= deltaN;
			q*= deltaQ;
		}
		// While exit flag hasnt been called by P0
		while (number != 1);
	}

	return 0;
}

/*
* Main Algorithm for 1D Distributed
*/
int* MM_1D_Distributed(const int* A_matrix, const int* B_matrix, const int m, const int n, const int q)
{
	// Variable Setup
	int pid;
	int p_total;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &p_total);
	// Master Thread
	if ( pid == 0 )
	{
		// Easier to deal with
		int* B_transpose = transposeMatrix(B_matrix, n, q);

		// ------ SCATTER ------
		// For each other processor...
		for (int this_pid = 1; this_pid < p_total; this_pid++)
		{
			// Send each process their segment of A
			int A_start = this_pid * m / p_total;
			int A_end = (this_pid + 1) * (m / p_total);
			int A_size = (A_end - A_start) * n;

			// Malloc number of rows * size of one row
			int* A_package = (int*)malloc(A_size * sizeof(int));

			// Rows of A
			for (int i = 0; i < A_size; i++)
			{
				// matrix is stored in row major order
				A_package[i] = A_matrix[(A_start * n) + i];
			}

			// MPI SEND
			MPI_Send(A_package, A_size, MPI_INT, this_pid, SCATTER_1D_A, MPI_COMM_WORLD);
			// Send each process their segment of B
			int B_start = this_pid * q / p_total;
			int B_end = (this_pid + 1) * (q / p_total);
			int B_size = (B_end - B_start) * n;

			// Malloc number of cols * size of one col
			int* B_package = (int*)malloc(B_size * sizeof(int));

			// Columns of B
			for (int i = 0; i < B_size; i++)
			{
				B_package[i] = B_transpose[(B_start * n) + i];
			}

			// MPI SEND
			MPI_Send(B_package, B_size, MPI_INT, this_pid, SCATTER_1D_B, MPI_COMM_WORLD);
		} // FREE PACKAGES (AND PANCAKES haha) ON EACH PROCESSOR, NOT HERE

		// ------ P0's MATRIX MULTIPY ------
		int* result = (int*)malloc(m * q * sizeof(int));

		int B_start = 0;
		int B_end = q / p_total;
		int B_size = (B_end - B_start) * n;
		int* P0_B_package = (int*)malloc(B_size * sizeof(int));

		// Pack P0's B
		for (int i = 0; i < B_size; i++)
		{
			P0_B_package[i] = B_transpose[(B_start * n) + i];
		}

		// Broadcast P0's B to other processors
		for ( int this_pid = 1; this_pid < p_total; this_pid++)
		{
			MPI_Send(P0_B_package, B_size, MPI_INT, this_pid, PASSING_1D_B, MPI_COMM_WORLD);
		}

		// If P0 already has all of B... do things need to send it back to P0?
		// Probobly not right?
		int A_size = (m / p_total);

		// Calculate P0's C
		// For Numbers of rows in A
		for ( int i = 0; i < A_size; i++)
		{
			for ( int j = 0; j < q; j++)
			{
				int sum = 0;
				for (int k = 0; k < n; k++)
				{
					// A[m][n] * B[n][q]
					sum += A_matrix[(i * n) + k] * B_matrix[(k * q) + j];
				}

				// C[m][q]
				result[(i * q) + j] = sum;
			}
		}


		// ------ GATHER ------
		// For each other processor...
		for (int this_pid = 1; this_pid < p_total; this_pid++)
		{
			// Recieve each section of C
			int C_start = this_pid * m / p_total;
			int C_end = (this_pid + 1) * (m / p_total);
			int C_size = (C_end - C_start) * q;

			// Package Size = number of rows of C * size of one row of C
			int* C_package = (int*)malloc(C_size * sizeof(int));
			MPI_Recv(C_package, C_size, MPI_INT, this_pid, GATHER_1D, MPI_COMM_WORLD, &status);

			// Take the entire package
			for (int i = 0; i < C_size; i++)
			{		
				result[(this_pid * C_size) + i] = C_package[i];

			}
			free(C_package);
		}

		free(P0_B_package);
		free(B_transpose);
		// Return Answer
		return result;
	}
	else
	{
		// A and B segments assigned to this processor passed in at start
		// REMINDER: B is transposed and is in row major order

		// For recieving and sending B_packages
		int B_start = pid * q / p_total;
		int B_end = (pid + 1) * (q / p_total);
		int B_size = B_end - B_start;
		int* B_package = (int*)malloc(B_size * n * sizeof(int));

		// Resulting C package
		int C_start = pid * m / p_total;
		int C_end = (pid + 1) * (m / p_total);
		int C_size = C_end - C_start;

		int* C_package = (int*)malloc(C_size * q * sizeof(int));
		// Columns of B are split into each processor
		for (int this_pid = 0; this_pid < p_total; this_pid++)
		{
			// If current processor
			if (this_pid == pid)
			{
				// Send to all other processors except P0
				for (int dest_pid = 1; dest_pid < pid; dest_pid++)
				{
					MPI_Send(B_matrix, B_size * n, MPI_INT, dest_pid, PASSING_1D_B, MPI_COMM_WORLD);
				}
				for (int dest_pid = pid + 1; dest_pid < p_total; dest_pid++)
				{
					MPI_Send(B_matrix, B_size * n, MPI_INT, dest_pid, PASSING_1D_B, MPI_COMM_WORLD);
				}

				// Number of Rows of C = Number of Rows from A
				for (int i = 0; i < C_size; i++)
				{
					// For each column of B
					for (int j = 0; j < B_size; j++)
					{
						int sum = 0;
						for (int k = 0; k < n; k++)
						{
							// USES B_MATRIX INSTEAD OF B_PACKAGE
							// REMINDER: B is transposed
							sum += A_matrix[(i * n) + k] * B_matrix[(j * n) + k];
						}
						// number of row * row size
						C_package[(i * m) + j + (this_pid * B_size)] = sum;
					}
				}
			}
			else
			{
				MPI_Recv(B_package, B_size * n, MPI_INT, this_pid, PASSING_1D_B, MPI_COMM_WORLD, &status);
				// Number of Rows of C = Number of Rows from A
				for (int i = 0; i < C_size; i++)
				{
					// For each column of B
					for (int j = 0; j < B_size; j++)
					{
						int sum = 0;
						for (int k = 0; k < n; k++)
						{
							// REMINDER: B is transposed
							sum += A_matrix[(i * n) + k] * B_package[(j * n) + k];
						}
						C_package[(i * m) + j + (this_pid * B_size)] = sum;
					}
				}
			}
		}

		// Send master thread the rows of C_package
		MPI_Send(C_package, C_size * q, MPI_INT, 0, GATHER_1D, MPI_COMM_WORLD);

		return NULL;
	}

	// Default return
	return NULL;
}

/*
* For testing or comparing
* Sequential plain version of matrix multiplication
*
* BEWARE USING WITH LARGE DIMENSIONS CAN TAKE HOURS IF NOT DAYS
*/
int* MM_sequential(const int* A, const int* B, const int m, const int n, const int q)
{
	// A = m x n, B = n x q, C = m x q
	int* result = (int*)malloc(m * q * sizeof(int));

	for ( int i = 0; i < m; i++)
	{
		for ( int j = 0; j < q; j++)
		{
			int sum = 0;
			for (int k = 0; k < n; k++)
			{
				// A[m][n] * B[n][q]
				sum += A[(i * n) + k] * B[(k * q) + j];
			}

			// C[m][q]
			result[(i * q) + j] = sum;
		}
	}

	return result;
}



/*
* Create a flattened 2d matrix of size m x n
*/
int* randomMatrix(const int m, const int n)
{
	// Empty Matrix size m x n
	int* result = (int*)malloc( m * n * sizeof(int));

	// Fill randomly with values 0 - 4
	for (int i = 0; i < m * n; i++)
	{
		result[i] = rand() % 5;
	}

	// Return randomized matrix
	return result;
}

/*
* Transpose the contents of a flattened 2d matrix of size m x n
*/
int* transposeMatrix(const int* A, const int m, const int n)
{
	int* result = (int*)malloc( n * m * sizeof(int));

	for ( int i = 0; i < m; i++)
	{
		for ( int j = 0; j < n; j++)
		{
			result[(i * m) + j] = A[(j * n) + i];
		}
	}
	return result;
}

/*
* Print the contents of a flattend matrix of size m x n
*/
void printMatrix(const int* A, const int m, const int n)
{
	for (int i = 0; i < m; i++)
	{
		for ( int j = 0; j < n; j++)
		{
			cout << A[(i * m) + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

/*
* Compare contents of two flattend matricies of size m x n
*/
int compareMatrix( const int* A, const int *B, const int m, const int n)
{
	for (int i = 0; i < m; i++)
	{
		for ( int j = 0; j < n; j++ )
		{
			// If difference found
			if ( A[(i * m) + n] != B[(i * m) + n])
				return 1;
		}
	}

	// No difference found
	return 0;
}

string appendString(const string original, const int value)
{
	string result = original + to_string(value);

	return result;
}

/*
 ============================================================================
 Name        : Algo.c
 Author      : Jeffrey A. Aborot
 Version     :
 Copyright   : This work is open-source but requires proper citation if to be used.
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

float* zero(){


}

int main(void) {
	puts("!!!Quantum algorithm C for approximate string matching!!!"); /* prints !!!Hello World!!! */

	/* Assumptions:
	 * 1. A qRAM holds a copy of T and can be queried for any i-th element of T
	 * 		by passing i as parameter value. qRAM puts its output register into
	 * 		the state representing the i-th symbol of T. i.e. qRAM^T(i) = T[i]
	 * 2. A qRAM holds data about the location of first occurrence in P of any
	 * 		symbol in Sigma. i.e. qRAM^P(T[i]) = location of first occurrence of
	 * 		symbol T[i] in P.
	 * 3. Sigma = {a, c, t, g}
	 */

	/* Algo C Steps:
	 * 1. Initialize.
	 * 	1.1 index register with log N bits into state 0 [accomodates
	 * 			integers 0,1,...,N-1]
	 * 	1.2 symbol register with log |Sigma| bits into state 0 [accomodates
	 * 			binary representation of symbols in Sigma]
	 * 	1.3 location register with log |Sigma| + 1 bit into state 0
	 * 			[accomodates integers -N,-(N-1),...,0,1,...,N-1]
	 * 2. Put into a superposition the index register.
	 * 3. Put the symbol register into the state |Beta(T[i])> where Beta(.) is
	 * 			a function which maps a symbol in Sigma to an log Sigma-bit
	 * 			binary number.
	 * 4. Put the location register into the state |Psi(T[i])> where Psi(.) is
	 * 			a function which maps a symbol in Sigma to its location of
	 * 			first occurrence in P.
	 * 5. Put the location register into the state |i-Psi(T[i])>.
	 * 6. Measure the state of the location register and output the classical
	 * 			value.
	 */

	// Alphabet
	char alphabet[4] = {'a', 'c', 't', 'g','\0'};
	int alphabet_size = sizeof(alphabet) * sizeof(char);
	printf("Alphabet: %s", alphabet);
	printf("Alphabet size: %d", alphabet_size);

	// Text
	printf("Specify text: ");
	char text[20];
	gets(text);

	// Step 1. Initialize
//	float index_register[log N];
//	float symbol_register[log |Sigma|];
//	float location_register[log N + 1];
//	float* index_register_ptr = zero(index_register);
//	float* symbol_register_ptr = zero(symbol_register);
//	float* location_register_ptr = zero(location_register);

	return EXIT_SUCCESS;
}

/* Read this comment first: https://gist.github.com/tonious/1377667#gistcomment-2277101
 * 2017-12-05
 * 
 *  -- T.
 */

#define _XOPEN_SOURCE 500 /* Enable certain library functions (strdup) on linux.  See feature_test_macros(7) */

#include "encoder_decoder.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>

struct entry_s {
	char *key;
	int value;
	struct entry_s *next;
};

typedef struct entry_s entry_t;

struct hashtable_s {
	int size;
	struct entry_s **table;	
};

typedef struct hashtable_s hashtable_t;


/* Create a new hashtable. */
hashtable_t *ht_create( int size ) {

	hashtable_t *hashtable = NULL;
	int i;

	if( size < 1 ) return NULL;

	/* Allocate the table itself. */
	if( ( hashtable = malloc( sizeof( hashtable_t ) ) ) == NULL ) {
		return NULL;
	}

	/* Allocate pointers to the head nodes. */
	if( ( hashtable->table = malloc( sizeof( entry_t * ) * size ) ) == NULL ) {
		return NULL;
	}
	for( i = 0; i < size; i++ ) {
		hashtable->table[i] = NULL;
	}

	hashtable->size = size;

	return hashtable;	
}

/* Hash a string for a particular hash table. */
int ht_hash( hashtable_t *hashtable, char *key ) {

	unsigned long int hashval;
	int i = 0;

	/* Convert our string to an integer */
	while( hashval < ULONG_MAX && i < strlen( key ) ) {
		hashval = hashval << 8;
		hashval += key[ i ];
		i++;
	}

	return hashval % hashtable->size;
}

/* Create a key-value pair. */
entry_t *ht_newpair( char *key, int value ) {
	entry_t *newpair;

	if( ( newpair = malloc( sizeof( entry_t ) ) ) == NULL ) {
		return NULL;
	}

	if( ( newpair->key = strdup( key ) ) == NULL ) {
		return NULL;
	}
    newpair->value = value;

	newpair->next = NULL;

	return newpair;
}

/* Insert a key-value pair into a hash table. */
void ht_set( hashtable_t *hashtable, char *key, int value ) {
	int bin = 0;
	entry_t *newpair = NULL;
	entry_t *next = NULL;
	entry_t *last = NULL;

	bin = ht_hash( hashtable, key );

	next = hashtable->table[ bin ];

	while( next != NULL && next->key != NULL && strcmp( key, next->key ) > 0 ) {
		last = next;
		next = next->next;
	}

	/* There's already a pair.  Let's replace that string. */
	if( next != NULL && next->key != NULL && strcmp( key, next->key ) == 0 ) {

		next->value = value;

	/* Nope, could't find it.  Time to grow a pair. */
	} else {
		newpair = ht_newpair( key, value );

		/* We're at the start of the linked list in this bin. */
		if( next == hashtable->table[ bin ] ) {
			newpair->next = next;
			hashtable->table[ bin ] = newpair;
	
		/* We're at the end of the linked list in this bin. */
		} else if ( next == NULL ) {
			last->next = newpair;
	
		/* We're in the middle of the list. */
		} else  {
			newpair->next = next;
			last->next = newpair;
		}
	}
}

/* Retrieve a key-value pair from a hash table. */
int ht_get( hashtable_t *hashtable, char *key ) {
	int bin = 0;
	entry_t *pair;

	bin = ht_hash( hashtable, key );

	/* Step through the bin, looking for our value. */
	pair = hashtable->table[ bin ];
	while( pair != NULL && pair->key != NULL && strcmp( key, pair->key ) > 0 ) {
		pair = pair->next;
	}

	/* Did we actually find anything? */
	if( pair == NULL || pair->key == NULL || strcmp( key, pair->key ) != 0 ) {
		return -1;

	} else {
		return pair->value;
	}
	
}

int  append(char *s, size_t size, char c) {
     if(strlen(s) + 1 >= size) {
          return 1;
     }
     int len = strlen(s);
     s[len] = c;
     s[len+1] = '\0';
     return 0;
}

int main( int argc, char **argv ) {
    int pix[] = {0, 1, 2}, i = 0;
    char *codes[3];
    char *encoded;
    int symbols[15];
    int img[7500];
    for(int i = 0; i < 15; i++){symbols[i] = i;}
	for(int i = 0; i < 7500; i++){img[i] = (i%15); }
    codes[0] = "0001";
    codes[1] = "0010";
    codes[2] = "0011";
    codes[3] = "0100";
	codes[4] = "0101";
	codes[5] = "0110";
	codes[6] = "0111";
	codes[7] = "1000";
	codes[8] = "1001";
    codes[9] = "1010";
    codes[10] = "1011";
    codes[11] = "1100";
	codes[12] = "1101";
	codes[13] = "1110";
	codes[14] = "1111";
	printf("Encoding.\n");
    encode(codes, img, 50, 50, 3, &encoded);
    printf("%s bits.\n", strlen(encoded));
		int *decoded = malloc(7500 * sizeof(int));
		for(int i = 0; i < 7500; i++){decoded[i] = -1;}

	printf("Decoding.\n");			
	
	decode(codes, symbols, encoded, decoded, 7500, 15);
    for(int i = 0; i < 7500; i++){ printf("%d ", decoded[i]);}
	printf("\n");
	return 0;
}

void encode(char** codes, int* img, int height, int width, int depth, char** encoded){
    size_t i, size, len = 0, size_str = sizeof(char)*15, nexp = 0;
    char *code, *buffer = NULL, *result = malloc(size_str), *current = NULL;
	ptrdiff_t diff;


    size = height * width * depth;
    if(!encoded){
        encoded = malloc(size * sizeof(int));
        if(!encoded){ printf("There's not enough memory!\n"); exit(1);}
    }

	result[0] = '\0';
	len = 1;
	for(i = 0; i < size; i++){
		code = codes[img[i]];
		len += strlen(code);
		if(len*sizeof(char) >= size_str - 10){
			size_str *= 2;
			result = realloc(result, size_str);
		}
		strcat(result, code);
	}
	*encoded = result;
	printf("%d bts.\n", len);
}

void decode(char** codes, int* symbols, char* code, int* decoded, int decodedsize, int nsymbols){
    hashtable_t *hashtable = ht_create(65536);
    size_t i = 0, j = 0, len = 0, codesize = strlen(code);
	char current_code[15];
	int value;

    if(decoded == NULL){
        decoded = malloc(decodedsize * sizeof(int));
        if(!decoded){ printf("Not enough memory!\n"); exit(1); }

        for(i = 0; i < decodedsize; i++){
            decoded[i] = -1;
        }
    }

    for(i = 0; i < nsymbols; i++){
        ht_set(hashtable, codes[i], symbols[i]);
    }
    
    current_code[0] = '\0';
    for(i = 0; i < codesize; i++){        
        append(current_code, codesize, code[i]);

        value = ht_get(hashtable, current_code);
		decoded[j] = value;
        if(value != -1){
            j++;
            if(j % decodedsize == 0){
                if(i < codesize-1)
                    printf("Decoded array is not big enough to hold the decoded values!\n");
                break;
            }
            current_code[0] = '\0';
        }
    }
}

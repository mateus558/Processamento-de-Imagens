// C program for Huffman Coding
#include <stdio.h>
#include <stdlib.h>
 
// This constant can be avoided by explicitly
// calculating height of Huffman Tree
#define MAX_TREE_HT 100
 
// A Huffman tree node
struct MinHeapNode {
 
    // One of the input characters
    char data;
 
    // Frequency of the character
    unsigned freq;
 
    // Left and right child of this node
    struct MinHeapNode *left, *right;
};
 
// A Min Heap:  Collection of
// min heap (or Hufmman tree) nodes
struct MinHeap {
 
    // Current size of min heap
    unsigned size;
 
    // capacity of min heap
    unsigned capacity;
 
    // Attay of minheap node pointers
    struct MinHeapNode** array;
};
 
// A utility function allocate a new
// min heap node with given character
// and frequency of the character
struct MinHeapNode* newNode(char data, unsigned freq)
{
    struct MinHeapNode* temp
        = (struct MinHeapNode*)malloc
(sizeof(struct MinHeapNode));
 
    temp->left = temp->right = NULL;
    temp->data = data;
    temp->freq = freq;
 
    return temp;
}
 
// A utility function to create
// a min heap of given capacity
struct MinHeap* createMinHeap(unsigned capacity)
 
{
 
    struct MinHeap* minHeap
        = (struct MinHeap*)malloc(sizeof(struct MinHeap));
 
    // current size is 0
    minHeap->size = 0;
 
    minHeap->capacity = capacity;
 
    minHeap->array
        = (struct MinHeapNode**)malloc(minHeap->
capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}
 
// A utility function to
// swap two min heap nodes
void swapMinHeapNode(struct MinHeapNode** a,
                     struct MinHeapNode** b)
 
{
 
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}
 
// The standard minHeapify function.
void minHeapify(struct MinHeap* minHeap, int idx)
 
{
 
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
 
    if (left < minHeap->size && minHeap->array[left]->
freq < minHeap->array[smallest]->freq)
        smallest = left;
 
    if (right < minHeap->size && minHeap->array[right]->
freq < minHeap->array[smallest]->freq)
        smallest = right;
 
    if (smallest != idx) {
        swapMinHeapNode(&minHeap->array[smallest],
                        &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}
 
// A utility function to check
// if size of heap is 1 or not
int isSizeOne(struct MinHeap* minHeap)
{
 
    return (minHeap->size == 1);
}
 
// A standard function to extract
// minimum value node from heap
struct MinHeapNode* extractMin(struct MinHeap* minHeap)
 
{
 
    struct MinHeapNode* temp = minHeap->array[0];
    minHeap->array[0]
        = minHeap->array[minHeap->size - 1];
 
    --minHeap->size;
    minHeapify(minHeap, 0);
 
    return temp;
}
 
// A utility function to insert
// a new node to Min Heap
void insertMinHeap(struct MinHeap* minHeap,
                   struct MinHeapNode* minHeapNode)
 
{
 
    ++minHeap->size;
    int i = minHeap->size - 1;
 
    while (i && minHeapNode->freq < minHeap->array[(i - 1) / 2]->freq) {
 
        minHeap->array[i] = minHeap->array[(i - 1) / 2];
        i = (i - 1) / 2;
    }
 
    minHeap->array[i] = minHeapNode;
}
 
// A standard funvtion to build min heap
void buildMinHeap(struct MinHeap* minHeap)
 
{
 
    int n = minHeap->size - 1;
    int i;
 
    for (i = (n - 1) / 2; i >= 0; --i)
        minHeapify(minHeap, i);
}
 
// A utility function to print an array of size n
void printArr(int arr[], int n)
{
    int i;
    for (i = 0; i < n; ++i)
        printf("%d", arr[i]);
 
    printf("\n");
}
 
// Utility function to check if this node is leaf
int isLeaf(struct MinHeapNode* root)
 
{
 
    return !(root->left) && !(root->right);
}
 
// Creates a min heap of capacity
// equal to size and inserts all character of
// data[] in min heap. Initially size of
// min heap is equal to capacity
struct MinHeap* createAndBuildMinHeap(char data[], int freq[], int size)
 
{
 
    struct MinHeap* minHeap = createMinHeap(size);
 
    for (int i = 0; i < size; ++i)
        minHeap->array[i] = newNode(data[i], freq[i]);
 
    minHeap->size = size;
    buildMinHeap(minHeap);
 
    return minHeap;
}
 
// The main function that builds Huffman tree
struct MinHeapNode* buildHuffmanTree(char data[], int freq[], int size)
 
{
    struct MinHeapNode *left, *right, *top;
 
    // Step 1: Create a min heap of capacity
    // equal to size.  Initially, there are
    // modes equal to size.
    struct MinHeap* minHeap = createAndBuildMinHeap(data, freq, size);
 
    // Iterate while size of heap doesn't become 1
    while (!isSizeOne(minHeap)) {
 
        // Step 2: Extract the two minimum
        // freq items from min heap
        left = extractMin(minHeap);
        right = extractMin(minHeap);
 
        // Step 3:  Create a new internal
        // node with frequency equal to the
        // sum of the two nodes frequencies.
        // Make the two extracted node as
        // left and right children of this new node.
        // Add this node to the min heap
        // '$' is a special value for internal nodes, not used
        top = newNode('$', left->freq + right->freq);
 
        top->left = left;
        top->right = right;
 
        insertMinHeap(minHeap, top);
    }
 
    // Step 4: The remaining node is the
    // root node and the tree is complete.
    return extractMin(minHeap);
}
 
// Prints huffman codes from the root of Huffman Tree.
// It uses arr[] to store codes
void printCodes(struct MinHeapNode* root, int arr[], int top)
 
{
 
    // Assign 0 to left edge and recur
    if (root->left) {
 
        arr[top] = 0;
        printCodes(root->left, arr, top + 1);
    }
 
    // Assign 1 to right edge and recur
    if (root->right) {
 
        arr[top] = 1;
        printCodes(root->right, arr, top + 1);
    }
 
    // If this is a leaf node, then
    // it contains one of the input
    // characters, print the character
    // and its code from arr[]
    if (isLeaf(root)) {
 
        printf("%c: ", root->data);
        printArr(arr, top);
    }
}
 
// The main function that builds a
// Huffman Tree and print codes by traversing
// the built Huffman Tree
void HuffmanCodes(char data[], int freq[], int size)
 
{
    // Construct Huffman Tree
    struct MinHeapNode* root
        = buildHuffmanTree(data, freq, size);
 
    // Print Huffman codes using
    // the Huffman tree built above
    int arr[MAX_TREE_HT], top = 0;
 
    printCodes(root, arr, top);
}
 
// Driver program to test above functions
int main()
{
 
    char arr[] = {211, 128, 255, 156, 171, 193, 238, 236, 30, 29, 33, 9, 7, 0, 2, 196, 116, 207, 108, 92, 87, 100, 99, 220, 8, 18, 17, 252, 209, 45, 49, 47, 173, 172, 35, 36, 44, 219, 153, 241, 84, 89, 157, 202, 203, 27, 26, 243, 242, 114, 46, 144, 158, 176, 230, 229, 1, 70, 251, 151, 107, 98, 57, 250, 248, 6, 62, 16, 80, 59, 20, 137, 138, 94, 118, 132, 147, 177, 42, 43, 86, 55, 4, 79, 76, 232, 249, 235, 22, 21, 75, 68, 240, 54, 237, 221, 32, 182, 191, 188, 34, 41, 200, 184, 103, 215, 10, 48, 120, 82, 73, 201, 124, 127, 109, 192, 194, 160, 161, 167, 19, 247, 3, 104, 51, 50, 244, 24, 23, 233, 13, 14, 95, 218, 217, 93, 83, 15, 12, 11, 63, 69, 53, 208, 40, 210, 245, 231, 212, 31, 28, 5, 254, 253, 37, 246, 227, 186, 125, 162, 65, 64, 152,
166, 214, 56, 146, 180, 183, 181, 77, 39, 225, 195, 58, 61, 105, 170, 25, 115, 150, 113, 163, 154, 169, 189, 216, 179, 239, 97, 234, 223, 168, 226, 206, 112, 71, 123, 204, 72, 178, 213, 67, 149, 164, 222, 141, 101, 142, 38, 145, 131, 91, 228, 119, 155, 185, 60, 224, 66, 88, 148, 139, 121, 197, 85, 110, 74, 135, 52, 205, 78, 90, 199, 122, 81, 165, 175, 174, 111,
140, 102, 159, 198, 130, 190, 143, 106, 136, 117};
    int freq[] = {7, 2, 266, 7, 5, 3, 34, 40, 13, 9, 3, 66, 47, 4083, 62, 5, 1, 7, 6, 4, 5, 6, 4, 20, 58, 53, 43, 36, 7, 6, 10, 10, 2, 8, 18, 13, 6, 8, 3, 24, 5, 6, 5, 12, 8, 11, 17, 46, 30, 4, 2, 2, 5, 4, 13, 21, 275, 5, 46, 2, 5, 5, 5, 54, 86, 47, 4, 25, 5, 7, 36, 5, 3, 6, 3, 3, 5, 2, 5, 6, 8, 5, 44, 3, 6, 29, 55, 28, 17, 26, 4, 3, 36, 4, 33, 17, 7, 4, 8, 3, 4, 12, 7, 4, 2, 6, 21, 11, 2, 3, 3, 9, 1, 1, 4, 3, 3, 2, 8, 4, 37, 67, 46, 2, 6, 3, 40, 27, 26, 24, 55, 43, 4, 11, 15, 2, 3, 15, 42, 29, 3, 4, 5, 6, 7, 7, 42, 17, 11, 13, 8, 45, 65, 34, 10, 28, 21, 6, 5, 3, 5, 5, 4, 3, 11, 3, 4, 7, 4, 3, 2, 7, 4, 4, 8, 3, 5, 4, 15, 3, 4, 1, 3, 4, 2, 3, 11, 5, 34, 2, 17, 8, 1, 21, 2, 2, 6, 1, 3, 2, 4, 7, 5, 2, 3, 4, 2, 3, 2, 4, 1, 1, 5, 15, 3, 1,
2, 4, 6, 2, 4, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 4, 2, 2, 1, 2, 3, 3, 1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 1, 1};
 
    int size = sizeof(arr) / sizeof(arr[0]);
 
    HuffmanCodes(arr, freq, size);
 
    return 0;
}

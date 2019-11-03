#include<stdio.h>



#define TILE_WIDTH 10

/*matrix multiplication kernels*/

//non shared
__global__ void MatrixMul( int *Md , int *Nd , int *Pd , const int WIDTH )
{

           // calculate thread id

           unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;

           unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

           Pd[row*WIDTH + col]=0;

         for (int k = 0 ; k<WIDTH ; k++ )
         {
                  Pd[row*WIDTH + col]+= Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;
          }

}


// main routine
int main ()
{
   const int WIDTH = 10 ;  //you can take large size as 100 200 300 400
   int array1_h[WIDTH][WIDTH] ,array2_h[WIDTH][WIDTH],result_array_h[WIDTH][WIDTH];
  int *array1_d , *array2_d ,*result_array_d ; // device array
  int i , j ;
  //input in host array
 // printf("Enter matrix1\n");
  for ( i = 0 ; i < WIDTH ; i++ )
  {
     for (j = 0 ; j < WIDTH ; j++ )
     {
        //scanf("%d",&array1_h[i][j]);
    	 array1_h[i][j]=rand()%10;
    	 array2_h[i][j]=rand()%10;
     }
  }


  //create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;

  cudaMalloc((void **) &array1_d , WIDTH*WIDTH*sizeof (int) ) ;

  cudaMalloc((void **) &array2_d , WIDTH*WIDTH*sizeof (int) ) ;




  //copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )

  cudaMemcpy ( array1_d , array1_h , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;

  cudaMemcpy ( array2_d , array2_h , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;



  //allocating memory for resultant device array

  cudaMalloc((void **) &result_array_d , WIDTH*WIDTH*sizeof (int) ) ;





  //calling kernal

  dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;

  dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;
  MatrixMul <<<dimGrid,dimBlock>>> ( array1_d , array2_d ,result_array_d , WIDTH) ;



  cudaMemcpy(result_array_h , result_array_d , WIDTH*WIDTH*sizeof(int) ,cudaMemcpyDeviceToHost) ;


  printf("Matrix 1\n");
  for ( i = 0 ; i <  WIDTH ; i++ )
   {
       for ( j = 0 ; j < WIDTH ; j++ )
      {
         printf ("%d   ",array1_h[i][j] ) ;
      }
  printf ("\n") ;
 }
  printf("\nMatrix 2\n");
  for ( i = 0 ; i < WIDTH ; i++ )
   {
       for ( j = 0 ; j < WIDTH ; j++ )
      {
         printf ("%d   ",array2_h[i][j] ) ;
      }
  printf ("\n") ;
 }
  printf("Matrix Multiplication Result\n");
  for ( i = 0 ; i < WIDTH ; i++ )
  {
      for ( j = 0 ; j < WIDTH ; j++ )
     {
        printf ("%d   ",result_array_h[i][j] ) ;
     }
 printf ("\n") ;
}
 return 0;
}
/*
**************************output*********************************************
Matrix 1
3   7   3   6   9   2   0   3   0   2
1   7   2   2   7   9   2   9   3   1
9   1   4   8   5   3   1   6   2   6
5   4   6   6   3   4   2   4   4   3
7   6   8   3   4   2   6   9   6   4
5   4   7   7   7   2   1   6   5   4
0   1   7   1   9   7   7   6   6   9
8   2   3   0   8   0   6   8   6   1
9   4   1   3   4   4   7   3   7   9
2   7   5   4   8   9   5   8   3   8

Matrix 2
6   5   5   2   1   7   9   6   6   6
8   9   0   3   5   2   8   7   6   2
3   9   7   4   0   6   0   3   0   1
5   7   5   9   7   5   5   7   4   0
8   8   4   1   9   0   8   2   6   9
0   8   1   2   2   6   0   1   9   9
9   7   1   5   7   6   3   5   3   4
1   9   9   8   5   9   3   5   1   5
8   8   0   0   4   4   6   1   5   6
1   8   7   1   5   7   3   8   1   9
Matrix Multiplication Result
190   278   145   132   190   136   200   169   161   167
186   355   156   157   207   209   185   164   210   246
191   335   233   179   196   257   220   227   174   232
191   319   172   156   167   218   182   186   165   186
276   433   239   205   229   305   251   252   193   257
233   378   222   181   218   240   231   216   180   226
232   430   221   155   255   274   187   203   193   328
248   319   178   137   201   217   233   171   165   236
267   379   184   141   231   276   259   247   218   301
252   477   239   204   282   302   239   261   245   334

*/

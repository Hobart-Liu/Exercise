/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package homework.ch1;

import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;

/**
 *
 * @author RDCLIYO
 */
public class ABC {
    
    public static void main(String[] args){
        int a = 250;
        float b = 400;
        float c = a/b;
        StdOut.println(c);


    }
    
    public static long F(int N){
        if (N==0) return 0;
        if (N==1) return 1;
        return F(N-1)+F(N-2);
    }
    
    public static void log2N(int num){
        if (num <= 1) 
            StdOut.println("1");
        int n = 2;
        int i = 0;
        while (n <= num){
            i++;
            n = n * 2;
        }
        StdOut.println(i);
        StdOut.println(Math.pow(2, i));
        
    }
    
    public static void transpose(){
        int[][] a = new int[2][3];
        for (int i=0; i<2 ; i++){
            for (int j= 0; j < 3; j++){
                a[i][j] = (i+1)*10+j;
            }
        }
        
        printMatrix(a);
        
        int[][] b = new int[a[0].length][a.length];
        
        for (int i = 0; i < a[0].length; i++){
            for (int j = 0; j < a.length; j++){
                b[i][j] = a[j][i];
            }
        }
        printMatrix(b);
       

            
    }
    
    public static void printMatrix(int[][] m){
        for (int i = 0; i < m.length; i++){
            for (int j = 0; j < m[0].length; j++){
                StdOut.printf("%2d ", m[i][j]);
            }
            StdOut.printf("\n");
        }
    }
    
    public static void binaryString(int num){
        String s = "";
        for (int n = num; n> 0; n/=2)
            s = (n%2) + s;
        StdOut.println(s);
    }
    
    
    
    public static void basiccmd(){
        double a = 1.0/0.0;
        double b = Double.POSITIVE_INFINITY;
        System.out.println(a);
        System.out.println(b);
        System.out.println(a==b);
        
        
        int x = 10;
        int y = 2;
        System.out.println(x&y);
        System.out.println(true && true || true && false);
        
        System.out.println(2.0E-6*(1.0E+8+0.1));   
        
        
        int aa = StdIn.readInt();
        int bb = StdIn.readInt();
        int cc = StdIn.readInt();
        if (aa==bb && bb == cc)
            StdOut.println("equal");
        
        
    }
    
}

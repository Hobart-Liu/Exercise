package assignment.assignment1;

import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.WeightedQuickUnionUF;
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdStats;


public class Percolation {

    private WeightedQuickUnionUF grid;
    private int n;
    private int top;
    private int bottom;
    private boolean[] stat;
    
    private int[][] deviate = {{0,1},{0, -1},{1,0},{-1,0}};
    
    
    public Percolation(int n){
        // create n-by-n grid, with all sites blocked
        this.n = n;
        grid = new WeightedQuickUnionUF(n*n+2);
        
        top = n*n;
        bottom = top + 1;
        
        stat = new boolean[n*n+2];
        for (int i = 0; i < n*n; i++){
            stat[i] = false;
        }
        stat[top] = true;
        stat[bottom] = true;
 
    }
    
    public void open(int i, int j){
        validate(i, j);
        // open site (row i, column j) if it is not open already
        int idx = this.getIndex(i, j);
        stat[idx] = true;
        
        int newx, newy, t, newidx;
        
        for (t = 0; t < 4; t++){
            newx = i + this.deviate[t][0];
            newy = j + this.deviate[t][1];
            if (newx > 0 && newx <=n && newy >0 && newy <= n){
                newidx = this.getIndex(newx, newy);
                if (stat[newidx]){
                    grid.union(idx, newidx);
                }
            }
        }
        
        if (i==1) grid.union(idx, top);
        if (i==n) grid.union(idx, bottom);
        
    }
    
    public boolean isOpen(int i, int j){
        this.validate(i, j);
        int idx = this.getIndex(i, j);
                
        return stat[idx];
    }
    
    public boolean isFull(int i, int j){
        this.validate(i, j);
        int idx = this.getIndex(i, j);
        return grid.connected(top, idx);
        
    }
    
    public boolean percolates(){
        return grid.connected(top, bottom);
    }
    
    private void validate(int i, int j) {
        if (i <= 0 || i > this.n || j <= 0 || j > this.n) {
            throw new IndexOutOfBoundsException("x = " + i + " y = " + j);  
        }
    }
    
    private int getIndex(int row, int col){
        // covert 2D dimension into 1D index
        return (row-1)*n + (col - 1);
    }
    
    
    
    
    public static void main(String[] args){
        
        int size = 20;
        Percolation p = new Percolation(size);
        
        int x,y, count = 0;
        
        while (!p.percolates()){
            x = StdRandom.uniform(1, size+1);
            y = StdRandom.uniform(1, size+1);
            if (!p.isOpen(x, y)){
                p.open(x, y);
                count++;
            }
        }
        StdOut.printf("After %d rounds\n", count);
        
    }
    
    
    
}

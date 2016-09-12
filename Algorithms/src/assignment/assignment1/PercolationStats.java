
package assignment.assignment1;

import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdStats;

public class PercolationStats {
    
    private double[] counter;
    private int n;
    
    
    public PercolationStats(int n, int trials){
        // perform trials independent experiments on an n-by-n grid
        counter = new double[trials];
        n = n;
        int x,y;
        float count;
        
        Percolation p;
        
        for (int i = 0; i < trials; i++){
            p = new Percolation(n);
            count = 0;
        
            while (!p.percolates()){
                x = StdRandom.uniform(1, n+1);
                y = StdRandom.uniform(1, n+1);
                if (!p.isOpen(x, y)){
                    p.open(x, y);
                    count++;
                }
            }
            counter[i] = count/(n*n);
            //StdOut.println(count+ "  " +  n*n + " " + counter[i] + " " + count/(n*n));            
        }
        
    }
    
    public double mean(){
        // sample mean of percolation threshold
        
        return StdStats.mean(counter);
    }
    
    public double stddev(){
        // sample standard deviation of percolation threshold
        return StdStats.stddev(counter);
    }
    
    public double confidenceLo(){
        // low endpoint of 95% confidence interval
        double ret = (mean()-1.96*stddev()/Math.sqrt(counter.length));
        //StdOut.println(ret);
        return ret;
    }
    
    public double confidenceHi(){
        // high endpoint of 95% confidence interval
        double ret = (mean()+1.96*stddev()/Math.sqrt(counter.length));
        //StdOut.println(ret);
        return ret;
    }
    
    public static void main(String[] args){
        // testing client
        int n1= 0, n2= 0;
        PercolationStats ps;
        n1 = Integer.parseInt(args[0]);
        n2 = Integer.parseInt(args[1]);
        ps = new PercolationStats(n1, n2);
        StdOut.printf("mean                     = %f\n", ps.mean());
        StdOut.printf("stddv                    = %f\n", ps.stddev());
        StdOut.printf("confidence interval      = %f %f\n", ps.confidenceLo(), ps.confidenceHi());
        
        
    }
    
}

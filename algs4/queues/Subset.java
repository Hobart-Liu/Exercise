import java.util.Iterator;
import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;


public class Subset {
    public static void main(String[] args) {
        
        int num = Integer.parseInt(args[0]);
        RandomizedQueue<String> q = new RandomizedQueue<String>();
        while (!StdIn.isEmpty()) {
            q.enqueue(StdIn.readString());
        }
        
        Iterator<String> i = q.iterator();
        
        int count = 0;
        while (count < num  && i.hasNext()) {
            StdOut.println(i.next());
            count++;
        }
        
    }
}

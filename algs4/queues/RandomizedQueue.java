
import java.util.Iterator;
import edu.princeton.cs.algs4.StdRandom;


public class RandomizedQueue<Item> implements Iterable<Item> {
    
    private Item[] q;
    private int n = 0;
    
    public RandomizedQueue() {
        q = (Item[]) new Object[1];
    }


    public boolean isEmpty() { return n == 0; }
    public int size() { return n; }
    
    private void resize(int capacity) {
        //StdOut.println("resizing to " + capacity);
        Item[] copy = (Item[]) new Object[capacity];
        for (int i = 0; i < n; i++)
            copy[i] = q[i];
        q = copy;
    }
    
    public void enqueue(Item item) {
        if (item != null) {
            if (n == q.length) resize(2*q.length);
            q[n++] = item;
        } else {
            throw new java.lang.NullPointerException();
        }
        
    }
    public Item dequeue() { 
        if (isEmpty()) throw new java.util.NoSuchElementException();
        int r = StdRandom.uniform(n);
        Item ret = q[r];
        for (int i = r; i < n-1; i++) {
            q[i] = q[i+1];
        }
        // q[n] = null;
        n = n - 1;
        if (n > 0 && n == q.length/4) resize(q.length/2);
        return ret;
    }
    public Item sample() {
        if (isEmpty()) throw new java.util.NoSuchElementException();
        int r = StdRandom.uniform(n);
        return q[r];
        
    }

    public Iterator<Item> iterator() {
        return (Iterator<Item>) new RandomizedQueueIterator();
    }
    
    private class RandomizedQueueIterator implements Iterator<Item> {
        
        private Item[] queue;
        private int i = n;
        public RandomizedQueueIterator() {
            queue = (Item[]) new Object[n];
            
            for (int x = 0; x < n; x++)
                queue[x] = q[x];
            
            StdRandom.shuffle(queue);
        }
        public boolean hasNext()    { return i > 0;             }
        public void remove()        { throw new java.lang.UnsupportedOperationException(); }
        public Item next() { 
            if (i > 0)
                return queue[--i];
            else
                throw new java.util.NoSuchElementException();
        }
        
    }
    
    public static void main(String[] args){
        RandomizedQueue<Integer> rq = new RandomizedQueue<Integer>();
        rq.isEmpty();
        rq.size();
        rq.enqueue(1);
        rq.dequeue();
        
    }
}

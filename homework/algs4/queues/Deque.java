import java.util.Iterator;


public class Deque<Item> implements Iterable<Item> {
    
    private class Node {
        private Item item = null;
        private Node prior = null;
        private Node next = null;
    }
    
    private Node head = null;
    private Node tail = null;
    private int size = 0;
    
    public Deque()              {  }
    public boolean isEmpty()    { return (size == 0); }
    public int size()           { return size; }
    
    public void addFirst(Item item) {
        // add the item to the front
        
        if (item != null) {
            Node oldhead = head;
            head = new Node();
            head.item = item;
            head.prior = null;
            if (oldhead != null) {
                head.next = oldhead;
                oldhead.prior = head;
            } else {
                // oldhead is null, this is empty 
                tail = head;
            }
            
            size = size + 1;

        } else {
            throw new java.lang.NullPointerException();
        }
    }    
        

    
    public void addLast(Item item) {
        // add the item to the end
        
        if (item != null) {
            Node oldtail = tail;
            tail = new Node();
            tail.item = item;
            tail.next = null;
            if (oldtail != null) {
                tail.prior = oldtail;
                oldtail.next = tail;
            } else {
                // this was empty
                head = tail;
            }
            
            size = size + 1;
            
        } else {
            throw new java.lang.NullPointerException();
        }
    }
    
    public Item removeFirst() {
        // remove and return the item from the front
        if (isEmpty()) throw new java.util.NoSuchElementException();
        Item item = head.item; 
        head = head.next;
        if (head != null) head.prior = null;
        else              tail = null;
        size = size - 1;
        return item;
    }
    
    public Item removeLast() {
        // remove and return the item from the end
        if (isEmpty()) throw new java.util.NoSuchElementException();
        Item item = tail.item;
        tail = tail.prior;
        if (tail != null) tail.next = null;
        else              head = null;
        size = size - 1;
        return item;
    }
    
    public Iterator<Item> iterator() {
        // return an iterator over items in order from front to end
        
        return (Iterator<Item>) new DequeIterator();
    }
    
    
    
    private class DequeIterator implements Iterator<Item> {
        private Node current = head;
        public boolean hasNext()    { return current != null; }
        public void remove()        { throw new java.lang.UnsupportedOperationException(); }
        public Item next() {
            if (current != null) {
                Item item = current.item;
                current = current.next;
                return item;
            } else {
                throw new java.util.NoSuchElementException();
            }
            
        
        }
    }
    
    public static void main(String[] args){
        
        Deque<Integer> deque = new Deque<Integer>();
         deque.isEmpty();
         deque.isEmpty();
         deque.isEmpty();
         deque.addFirst(3);
         deque.removeLast();
         deque.isEmpty();
         deque.addFirst(6);
         deque.removeLast();

    }
            
    
}

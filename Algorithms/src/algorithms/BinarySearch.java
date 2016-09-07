package algorithms;


public class BinarySearch {
    

    /**
     * This class should not be instantiated.
     */
    private BinarySearch() { }
    
    /**
     * Returns the index of the specified key in the specified array.
     *
     * @param  a the array of integers, must be sorted in ascending order
     * @param  key the search key
     * @return index of key in array {@code a} if present; {@code -1} otherwise
     */
    public static int indexOf(int[] a, int key) {
        int lo = 0;
        int hi = a.length - 1;
        while (lo <= hi) {
            // Key is in a[lo..hi] or not present.
            int mid = lo + (hi - lo) / 2;
            if      (key < a[mid]) hi = mid - 1;
            else if (key > a[mid]) lo = mid + 1;
            else return mid;
        }
        return -1;
    }
    
    /**
     *  递归调用, same function as indexOf
     */
    public static int indexOf2(int[] a, int key){
        return indexOf2(key, a, 0, a.length-1);
    }
    
    public static int indexOf2(int key, int[] a, int lo, int hi){
        if (lo > hi) return -1;
        int mid = lo +(hi-lo)/2;
        if      (key < a[mid])  return indexOf2(key, a, lo, mid-1);
        else if (key > a[mid])  return indexOf2(key, a, mid+1, hi);
        else                    return mid;
    }
    
    
    /**
     * Simple Test
     * @param args: Not used.
     */
    
    public static void main(String[] args){
        int[] a = {15, 47, 30, 87, 40, 90, 92, 12, 63, 56, 10, 78, 62, 55};
        int[] b = {30, 87, 15, 55, 92, 77, 81, 48, 47, 62, 40,70};
        
        java.util.Arrays.sort(a);
        
        for(int i = 0; i < b.length; i++){
            if(BinarySearch.indexOf2(a, b[i])== -1)
                System.out.println(b[i]);
        }
                ;
        
    }
    
}

package Penny;

public class SlidingWindow {
	
	public double[] window;
	int ct = 0;
	int windowsize;
	
	public  SlidingWindow(int size){	
		windowsize = size;
		window = new double[windowsize];
   }
	
    public void put(double i){
    	window[ct % window.length] = i;
        ct++;
   }
    public double get(int index){
    	return window[index];
   }
    

}

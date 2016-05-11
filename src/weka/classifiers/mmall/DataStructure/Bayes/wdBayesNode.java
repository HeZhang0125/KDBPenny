package weka.classifiers.mmall.DataStructure.Bayes;

/*
* wdBayesNode - Node of Tree, storing discriminative parameters
*  
* dParameterTree constitutes of wdBayesNodes
*
*/
public class wdBayesNode {

	public double[] xyParameter;  	// Parameter indexed by x val the y val
	public double[] xyCount;		// Count for x val and the y val
	public double[] xyProbability;		// Count for x val and the y val

	public wdBayesNode[] children;	

	public int att;          // the Attribute whose values select the next child
	public int index;
	public int paramsPerAttVal;

	public int scheme;

	// default constructor - init must be called after construction
	public wdBayesNode(int s1) {
		scheme = s1;
	}     

	// Initialize a new uninitialized node
	public void init(int nc, int paramsPerAttVal) {
		this.paramsPerAttVal = paramsPerAttVal;
		index = -1;
		att = -1;

		if (scheme == 1) {
			xyCount = new double[nc * paramsPerAttVal];
			xyProbability = new double[nc * paramsPerAttVal];
		}  else if (scheme == 2) {
			xyParameter = new double[nc * paramsPerAttVal];
			xyCount = new double[nc * paramsPerAttVal];
			xyProbability = new double[nc * paramsPerAttVal];
		}			
		children = null;
	}  

	// Reset a node to be empty
	public void clear() { 
		att = -1;
		xyParameter = null;
		children = null;
	}      

	public void setXYParameter(int v, int y, double val) {
		xyParameter[y * paramsPerAttVal + v] = val;
	}

	public double getXYParameter(int v, int y) {
		return xyParameter[y * paramsPerAttVal + v];		
	}

	public void setXYCount(int v, int y, double val) {
		xyCount[y * paramsPerAttVal + v] = val;
	}

	public double getXYCount(int v,int y) {
		return xyCount[y * paramsPerAttVal + v];		
	}
	
	public void setXYProbability(int v, int y, double val) {
		xyProbability[y * paramsPerAttVal + v] = val;
	}

	public double getXYProbability(int v, int y) {
		return xyProbability[y * paramsPerAttVal + v];		
	}

	public int getXYIndex(int v, int y) {
		return index + (y * paramsPerAttVal + v);
	}

	public void incrementXYCount(int v, int y) {
		xyCount[y * paramsPerAttVal + v]++;
	}
	public void getProbability(){
		double p = 0.0;
	}
}
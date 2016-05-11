package weka.classifiers.mmall.Ensemble.generalization;

import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public abstract class wdAnJEParameters {

	protected long np;
	protected int npi;

	protected int n;
	protected int nc;
	protected int N;
	protected int scheme;

	protected int[] paramsPerAtt;	

	protected indexTrie[] indexTrie_;

	protected int [] xyCount;
	protected double [] probs;
	protected double [] parameters;
	protected double [] gradients;

	protected int numTuples;

	protected static int MAX_TAB_LENGTH = Integer.MAX_VALUE-8;
	//	protected static int MAX_TAB_LENGTH = 2009;
	protected double PARAMETER_VALUE_WHEN_ZERO_COUNT = 0.0;
	
	/**
	 * Constructor called by wdBayes
	 */
	public wdAnJEParameters(int n, int nc, int N, int[] in_ParamsPerAtt, int numTuples) {
		this.n = n;
		this.nc = nc;
		this.N = N;
		this.numTuples = numTuples;
		
		paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = in_ParamsPerAtt[u];
		}

		indexTrie_ = new indexTrie[n];				

		if (numTuples == 1) {
			np = nc;
			npi = nc;
			for (int u1 = 0; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();

				indexTrie_[u1].set(np);
				indexTrie_[u1].setIndex(npi);
				npi++;
				
				np += (paramsPerAtt[u1] * nc);
			}
		} else if (numTuples == 2) {
			np = nc;
			npi = nc;
			for (int u1 = 1; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 0; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].set(np);
					
					indexTrie_[u1].children[u2].setIndex(npi);
					npi++;

					np += (paramsPerAtt[u1] * paramsPerAtt[u2] * nc);												
				}					
			}
		} else if (numTuples == 3) {
			np = nc;
			npi = nc;
			for (int u1 = 2; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 1; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].children = new indexTrie[n];

					for (int u3 = 0; u3 < u2; u3++) {
						indexTrie_[u1].children[u2].children[u3] = new indexTrie();
						indexTrie_[u1].children[u2].children[u3].set(np);
						
						indexTrie_[u1].children[u2].children[u3].setIndex(npi);
						npi++;

						np += (paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * nc);												
					}					
				}
			}
		} else if (numTuples == 4) {
			np = nc;
			npi = nc;
			for (int u1 = 3; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 2; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].children = new indexTrie[n];

					for (int u3 = 1; u3 < u2; u3++) {
						indexTrie_[u1].children[u2].children[u3] = new indexTrie();
						indexTrie_[u1].children[u2].children[u3].children = new indexTrie[n];

						for (int u4 = 0; u4 < u3; u4++) {
							indexTrie_[u1].children[u2].children[u3].children[u4] = new indexTrie();
							indexTrie_[u1].children[u2].children[u3].children[u4].set(np);
							
							indexTrie_[u1].children[u2].children[u3].children[u4].setIndex(npi);
							npi++;

							np += (paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4] * nc);												
						}				
					}
				}
			}
		} else if (numTuples == 5) {
			np = nc;	
			npi = nc;
			for (int u1 = 4; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 3; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].children = new indexTrie[n];

					for (int u3 = 2; u3 < u2; u3++) {
						indexTrie_[u1].children[u2].children[u3] = new indexTrie();
						indexTrie_[u1].children[u2].children[u3].children = new indexTrie[n];

						for (int u4 = 1; u4 < u3; u4++) {
							indexTrie_[u1].children[u2].children[u3].children[u4] = new indexTrie();
							indexTrie_[u1].children[u2].children[u3].children[u4].children = new indexTrie[n];

							for (int u5 = 0; u5 < u4; u5++) {
								indexTrie_[u1].children[u2].children[u3].children[u4].children[u5] = new indexTrie();
								indexTrie_[u1].children[u2].children[u3].children[u4].children[u5].set(np);
								
								indexTrie_[u1].children[u2].children[u3].children[u4].children[u5].setIndex(npi);
								npi++;

								np += (paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4] * paramsPerAtt[u5] * nc);
							}
						}				
					}
				}
			}
		}

	}

	/**
	 * Function called in the first pass to look at the combinations that have been seen or not. 
	 * Then the function finishedUpdatingSeenObservations should be called, and then the update_MAP function. 
	 * @param inst
	 */
	public abstract void updateFirstPass(Instance inst);

	/**
	 * Multi-threaded version of updateFirstPass 
	 * @param inst
	 */
	public abstract void updateFirstPass_m(Instances m_Instances);

	/**
	 * Function called when the first pass is finished
	 */
	public abstract void finishedFirstPass();

	public abstract boolean needSecondPass();

	/**
	 * Function called to initialize the counts, if needed, in the second pass after having called 'update_seen_observations' on every instance first.
	 * Needs to be overriden, or will just do nothing 
	 * @param inst
	 */
	public void update_MAP(Instance inst) {

	}

	/**
	 * Multi-threaded version of update_MAP 
	 * @param inst
	 */
	public void update_MAP_m(Instances m_Instances) {

	}

	public abstract int getCountAtFullIndex(long index);
	public abstract void setCountAtFullIndex(long index,int count);
	public void incCountAtFullIndex(long index){
		incCountAtFullIndex(index,1);
	}
	public abstract void incCountAtFullIndex(long index,int value);

	public abstract void setProbAtFullIndex(long index,double p);
	public abstract double getProbAtFullIndex(long index);

	public abstract void setGradientAtFullIndex(long index,double g);
	public abstract double getGradientAtFullIndex(long index);
	public abstract void incGradientAtFullIndex(long index, double g);

	/**
	 * Set the value of one dimension of a given gradient
	 * @param gradient the gradient (array) to which the value has to be set
	 * @param index the index given as a coordinate in the full 1-d array
	 * @param value the value to which the specific dimension of the gradient has to be set to
	 */
	public abstract void setGradientAtFullIndex(double[]gradient,long index,double value);

	/**
	 * Get the value of one dimension of a given gradient
	 * @param gradient the gradient (array) from which the value has to be gotten
	 * @param index the index given as a coordinate in the full 1-d array
	 */
	public abstract double getGradientAtFullIndex(double[]gradient,long index);

	/**
	 * Increase the value of one dimension of a given gradient
	 * @param gradient the gradient (array) to which the value has to be set
	 * @param index the index given as a coordinate in the full 1-d array
	 * @param value the value by which the specific dimension of the gradient has to be increased 
	 */
	public abstract void incGradientAtFullIndex(double []gradient,long index, double value);


	public void convertToProbs() {

		for (int c = 0; c < nc; c++) {
			setProbAtFullIndex(c,  Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(c), N, nc), 1e-75)));
		}


		if (numTuples == 1) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 0; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						long index = getAttributeIndex(u1, u1val, c);
						setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1]), 1e-75)));
					}
				}				
			}

		} else if (numTuples == 2) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 1; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 0; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

								long index = getAttributeIndex(u1, u1val, u2, u2val, c);
								setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2]), 1e-75)));
							}
						}
					}
				}
			}	

		} else if (numTuples == 3) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 2; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 1; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 0; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {	

										long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
										setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3]), 1e-75)));
									}
								}
							}
						}
					}
				}
			}

		} else if (numTuples == 4) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 3; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 2; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 1; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 0; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
												setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4]), 1e-75)));
											}
										}
									}
								}
							}
						}
					}
				}
			}
		} else if (numTuples == 5) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 4; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 3; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 2; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 1; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												for (int u5 = 0; u5 < u4; u5++) {
													for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {

														long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
														setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4] * paramsPerAtt[u5]), 1e-75)));
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		xyCount = null;
		System.gc();
	}

	
	// ----------------------------------------------------------------------------------
	// Access Functions
	// ----------------------------------------------------------------------------------

	public abstract double getParameterAtFullIndex(long index);
	public abstract void setParameterAtFullIndex(long index,double p);

	public double[]getParameters(){
		return parameters;
	}

	public double[]getGradients(){
		return gradients;
	}

	public int getClassIndex(int k) {
		return k;
	}

	public long getAttributeIndex(int att1, int att1val, int c) {
		long offset = indexTrie_[att1].offset;		
		return offset + c * (paramsPerAtt[att1]) + att1val;
	}
	
	public int getAttributeIndexIndex(int att1) {
		return indexTrie_[att1].index;		
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int c) {
		long offset = indexTrie_[att1].children[att2].offset;		
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}
	
	public int getAttributeIndexIndex(int att1, int att2) {
		return indexTrie_[att1].children[att2].index;		
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].offset;
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) + 
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}
	
	public int getAttributeIndexIndex(int att1, int att2, int att3) {
		return indexTrie_[att1].children[att2].children[att3].index;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int att4, int att4val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].children[att4].offset;
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4]) +
				att4val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) +
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}
	
	public int getAttributeIndexIndex(int att1, int att2, int att3, int att4) {
		return indexTrie_[att1].children[att2].children[att3].children[att4].index;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int att4, int att4val, int att5, int att5val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].children[att4].children[att5].offset;		
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4] * paramsPerAtt[att5]) +
				att5val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4]) +
				att4val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) +
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}
	
	public int getAttributeIndexIndex(int att1, int att2, int att3, int att4, int att5) {
		return indexTrie_[att1].children[att2].children[att3].children[att4].children[att5].index;		
	}

	public double getParamsPetAtt(int att) {
		return paramsPerAtt[att];
	}

	public int getNumberParametersAllocated(){
		return parameters.length;
	}

	public long getTotalNumberParameters() {
		return np;
	}

	public int getTotalNumberParametersLevel() {
		return npi;
	}

	// ----------------------------------------------------------------------------------
	// Flat structure into array
	// ----------------------------------------------------------------------------------

	//public void copyParameters(double[] inParameters) {
	//	System.arraycopy(inParameters, 0, parameters, 0, np);
	//}

	public void copyParameters(double[] params) {
		System.arraycopy(params, 0, parameters, 0, params.length);
	}

	protected abstract void initCount(long size);

	protected abstract void initProbs(long size);

	protected abstract void initParameters(long size);

	protected abstract void initGradients(long size);

	public abstract void resetGradients();

	public int getNAttributes(){
		return n;
	}

	public void updateAfterFirstPass(Instance inst) {
		// TODO Auto-generated method stub
		
	}

	public void updateAfterFirstPass_m(Instances m_Instances) {
		// TODO Auto-generated method stub
		
	}

} // ends class


class indexTrie {

	public long offset;
	public int index;
	public indexTrie[] children;

	public indexTrie() {
		offset = -1;
		children = null;
	}	

	public void set(long val) {
		offset = val;
	}
	
	public void setIndex(int val) {
		index = val;
	}

	public long get() {
		return offset;
	}
	
	public int getIndex() {
		return index;
	}

}

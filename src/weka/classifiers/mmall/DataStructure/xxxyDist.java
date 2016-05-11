/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * xxxyDist.java     
 * Code written by: Nayyar Zaidi
 * 
 */

package weka.classifiers.mmall.DataStructure;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class xxxyDist {

	private double[][][][] counts_;
	private double[][][][] probs_;
	
	public xxyDist xxyDist_;

	private int N;
	private int n;
	private int nc;

	private int paramsPerAtt[];

	public xxxyDist(Instances instances) {

		N = instances.numInstances();
		n = instances.numAttributes() - 1; // -1 is due to the class presence in numAttributes
		nc = instances.numClasses();

		paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = instances.attribute(u).numValues();
		}

		xxyDist_ = new xxyDist(instances);
		counts_ = new double[n][][][];

		for (int u1 = 2; u1 < n; u1++) {
			counts_[u1] = new double[paramsPerAtt[u1] * u1][][];

			for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
				for (int u2 = 1; u2 < u1; u2++) {

					int pos1 = u1*u1val + u2;
					counts_[u1][pos1] = new double[paramsPerAtt[u2] * u2][];

					for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
						for (int u3 = 0; u3 < u2; u3++) {

							int pos2 = u2*u2val + u3;
							counts_[u1][pos1][pos2] = new double[paramsPerAtt[u3] * nc];							
						}
					}
				}
			}
		}

	}
	
	public void addToCount_m(Instances instances) {
		
		int nThreads;
		int minNPerThread = 10000;					
		int N = instances.numInstances();

		ExecutorService executor;

		if (N < minNPerThread) {
			nThreads = 1;
		} else {
			nThreads = Runtime.getRuntime().availableProcessors();
			if (N/nThreads < minNPerThread) {
				nThreads = N/minNPerThread + 1;
			}
		}
		System.out.println("In xxxyDist() - Pass 1: Launching " + nThreads + " threads");
				
		int[][][][][] threadXYCount = new int[nThreads][n][][][];

		for (int th = 0; th < nThreads; th++ ) {
			
			for (int u1 = 2; u1 < n; u1++) {
				threadXYCount[th][u1] = new int[paramsPerAtt[u1] * u1][][];

				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
					for (int u2 = 1; u2 < u1; u2++) {

						int pos1 = u1*u1val + u2;
						threadXYCount[th][u1][pos1] = new int[paramsPerAtt[u2] * u2][];

						for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
							for (int u3 = 0; u3 < u2; u3++) {

								int pos2 = u2*u2val + u3;
								threadXYCount[th][u1][pos1][pos2] = new int[paramsPerAtt[u3] * nc];							
							}
						}
					}
				}
			}
			
		}
		
		executor = Executors.newFixedThreadPool(nThreads);					

		Future<Double>[] futures = new Future[nThreads];

		int assigned = 0;
		int remaining = N;

		for (int th = 0; th < nThreads; th++) {
			/*
			 * Compute the start and stop indexes for thread th
			 */
			int start = assigned;
			int nInstances4Thread = remaining / (nThreads - th);
			assigned += nInstances4Thread;
			int stop = assigned - 1;
			remaining -= nInstances4Thread;

			/*
			 * Calling thread
			 */
			Callable<Double> thread = new CallableXXXYDist(start, stop, instances, threadXYCount[th], th);

			futures[th] = executor.submit(thread);
		}
		
		for (int th = 0; th < nThreads; th++) {
			
			try {
				double temp = futures[th].get();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}						
			
			//for (int i = 0; i < xyCount.length; i++) {
			//	xyCount[i] += threadXYCount[th][i];
			//}					
		}
		
		executor.shutdown();
		System.out.println("In xxxyDist() - Pass 1: All threads finished.");		
	}

	public void addToCount(Instances instances) {
		for (int ii = 0; ii < N; ii++) {
			Instance inst = instances.instance(ii);
			//xxyDist_.update(inst);

			update(inst);
		}
	}

	public void update(Instance inst) {
		xxyDist_.update(inst);

		int x_C = (int) inst.classValue();

		for (int u1 = 2; u1 < n; u1++) {
			int x_u1 = (int) inst.value(u1);

			for (int u2 = 1; u2 < u1; u2++) {
				int x_u2 = (int) inst.value(u2);

				for (int u3 = 0; u3 < u2; u3++) {
					int x_u3 = (int) inst.value(u3);

					int pos1 = x_u1*u1 + u2;
					int pos2 = x_u2*u2 + u3; 
					int pos3 = x_u3*nc + x_C;					

					counts_[u1][pos1][pos2][pos3]++;					
				}
			}
		}
	}

	public void countsToAJEProbs() {
		xxyDist_.countsToAJEProbs();
		
		probs_ = new double[n][][][];

		for (int u1 = 2; u1 < n; u1++) {
			probs_[u1] = new double[paramsPerAtt[u1] * u1][][];

			for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
				for (int u2 = 1; u2 < u1; u2++) {

					int pos1 = u1*u1val + u2;
					probs_[u1][pos1] = new double[paramsPerAtt[u2] * u2][];

					for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
						for (int u3 = 0; u3 < u2; u3++) {

							int pos2 = u2*u2val + u3;
							probs_[u1][pos1][pos2] = new double[paramsPerAtt[u3] * nc];							
						}
					}
				}
			}
		}

		for (int c = 0; c < nc; c++) {

			for (int u1 = 2; u1 < n; u1++) {
				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

					for (int u2 = 1; u2 < u1; u2++) {
						for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

							for (int u3 = 0; u3 < u2; u3++) {
								for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

									int pos1 = u1val*u1 + u2;
									int pos2 = u2val*u2 + u3; 
									int pos3 = u3val*nc + c;	

									probs_[u1][pos1][pos2][pos3] = Math.max(SUtils.MEsti(ref(u1,u1val,u2,u2val,u3,u3val,c), 
											xxyDist_.xyDist_.getClassCount(c), 
											paramsPerAtt[u1]*paramsPerAtt[u2]*paramsPerAtt[u3]), 1e-75);
								}
							}
						}
					}

				}
			}
		}
	}

	// p(x1=v1, x2=v2, x3=v3, Y=y) unsmoothed
	public double rawJointP(int x1, int v1, int x2, int v2, int x3, int v3, int y){
		return ref(x1, v1, x2, v2, x3, v3, y) / N;
	}

	// p(x1=v1, x2=v2, x3=v3, Y=y) using M-estimate
	public double jointP(int x1, int v1, int x2, int v2, int x3, int v3, int y) {
		//return (*constRef(x1, v1, x2, v2, x3, v3, y) + M / (getNoValues(x1) * getNoValues(x2) * getNoValues(x3) * noClasses_)) / (xxyCounts.xyCounts.count + M);
		return SUtils.MEsti(ref(x1, v1, x2, v2, x3, v3, y), N, paramsPerAtt[x1] * paramsPerAtt[x2] * paramsPerAtt[x3] * nc);
	}

	// p(x1=v1, x2=v2, x3=v3) using M-estimate
	public double jointP(int x1, int v1, int x2, int v2, int x3, int v3) {
		//return (getCount(x1, v1, x2, v2, x3, v3) + M / (metaData_->getNoValues(x1) * metaData_->getNoValues(x2)* metaData_->getNoValues(x3))) / (xxyCounts.xyCounts.count + M);
		return SUtils.MEsti(getCount(x1, v1, x2, v2, x3, v3), N, paramsPerAtt[x1] * paramsPerAtt[x2] * paramsPerAtt[x3]);
	}

	// p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
	public double p(int x1, int v1, int x2, int v2, int x3, int v3, int y) {
		//return (*constRef(x1, v1, x2, v2, x3, v3, y) + M / getNoValues(x1)) / (xxyCounts.getCount(x2, v2, x3, v3, y) + M);
		return SUtils.MEsti(ref(x1, v1, x2, v2, x3, v3, y), xxyDist_.getCount(x2, v2, x3, v3, y), paramsPerAtt[x1]);
	}
	
	// p(x1=v1, x2=v2, x3=v3|Y=y) using M-estimate, probabilities already computed
	public double jp(int x1, int v1, int x2, int v2, int x3, int v3, int y) {
		return jref(x1, v1, x2, v2, x3, v3, y);
	}

	// p(x1=v1, x2=v2, x3=v3, Y=y)
	public double getCount(int x1, int v1, int x2, int v2, int x3, int v3, int y) {
		return ref(x1, v1, x2, v2, x3, v3, y);
	}

	// count for instances x1=v1, x2=v2,x3=v3
	public int getCount(int x1, int v1, int x2, int v2, int x3, int v3) {
		int c = 0;

		for (int y = 0; y < nc; y++) {
			c += getCount(x1, v1, x2, v2, x3, v3, y);
		}
		return c;
	}

	// count[X1=x1][X2=x2][X3=x3][Y=y]
	private double ref(int x1, int v1, int x2, int v2, int x3, int v3, int y) {

		if (x2 > x1) {
			int t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}
		if (x3 > x2) {
			int t = x2;
			x2 = x3;
			x3 = t;
			t = v2;
			v2 = v3;
			v3 = t;
		}
		if (x2 > x1) {
			int t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}
		assert(x1 > x2 && x2 > x3);
		//return &count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * noClasses_ + y];

		int pos1 = v1*x1 + x2;
		int pos2 = v2*x2 + x3;
		int pos3 = v3*nc + y;

		return counts_[x1][pos1][pos2][pos3];
	}
	
	// probs[X1=x1][X2=x2][X3=x3][Y=y]
	private double jref(int x1, int v1, int x2, int v2, int x3, int v3, int y) {

		if (x2 > x1) {
			int t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}
		if (x3 > x2) {
			int t = x2;
			x2 = x3;
			x3 = t;
			t = v2;
			v2 = v3;
			v3 = t;
		}
		if (x2 > x1) {
			int t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}
		assert(x1 > x2 && x2 > x3);
		//return &count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * noClasses_ + y];

		int pos1 = v1*x1 + x2;
		int pos2 = v2*x2 + x3;
		int pos3 = v3*nc + y;

		return probs_[x1][pos1][pos2][pos3];
	}	

	public int getNoAtts() { return n; }

	public int getNoCatAtts() { return n; }

	public int getNoValues(int a) { return paramsPerAtt[a]; }

	public int getNoData() { return N; }

	public int getNoClasses() { return nc; }

	public int[] getNoValues() { return paramsPerAtt; }

}

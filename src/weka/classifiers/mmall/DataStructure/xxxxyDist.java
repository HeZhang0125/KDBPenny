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
 * xxxxyDist.java     
 * Code written by: Nayyar Zaidi
 * 
 */

package weka.classifiers.mmall.DataStructure;

import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class xxxxyDist {

	private double[][][][][] counts_;
	private double[][][][][] probs_;
	
	public xxxyDist xxxyDist_;

	private int N;
	private int n;
	private int nc;

	private int paramsPerAtt[];

	public xxxxyDist(Instances instances) {

		N = instances.numInstances();
		n = instances.numAttributes() - 1; // -1 is due to the class presence in numAttributes
		nc = instances.numClasses();

		paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = instances.attribute(u).numValues();
		}

		xxxyDist_ = new xxxyDist(instances);
		counts_ = new double[n][][][][];

		for (int u1 = 3; u1 < n; u1++) {
			counts_[u1] = new double[paramsPerAtt[u1] * u1][][][];

			for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
				for (int u2 = 2; u2 < u1; u2++) {					
					int pos1 = u1*u1val + u2;
					counts_[u1][pos1] = new double[paramsPerAtt[u2] * u2][][];

					for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
						for (int u3 = 1; u3 < u2; u3++) {
							int pos2 = u2*u2val + u3;
							counts_[u1][pos1][pos2] = new double[paramsPerAtt[u3] * u3][];

							for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {
								for (int u4 = 0; u4 < u3; u4++) {
									int pos3 = u3*u3val + u4;
									counts_[u1][pos1][pos2][pos3] = new double[paramsPerAtt[u4] * nc];
								}
							}
						}
					}
				}
			}
		}
	}

	public void addToCount(Instances instances) {
		for (int ii = 0; ii < N; ii++) {
			Instance inst = instances.instance(ii);
			//xxxyDist_.update(inst);

			update(inst);
		}
	}

	public void update(Instance inst) {
		xxxyDist_.update(inst);

		int x_C = (int) inst.classValue();

		for (int u1 = 3; u1 < n; u1++) {
			int x_u1 = (int) inst.value(u1);

			for (int u2 = 2; u2 < u1; u2++) {
				int x_u2 = (int) inst.value(u2);

				for (int u3 = 1; u3 < u2; u3++) {
					int x_u3 = (int) inst.value(u3);

					for (int u4 = 0; u4 < u3; u4++) {
						int x_u4 = (int) inst.value(u4);

						int pos1 = u1*x_u1 + u2;
						int pos2 = u2*x_u2 + u3;
						int pos3 = u3*x_u3 + u4;
						int pos4 = x_u4*nc + x_C;

						counts_[u1][pos1][pos2][pos3][pos4]++;						
					}
				}
			}
		}
	}

	public void countsToAJEProbs() {
		xxxyDist_.countsToAJEProbs();
		
		probs_ = new double[n][][][][];

		for (int u1 = 3; u1 < n; u1++) {
			probs_[u1] = new double[paramsPerAtt[u1] * u1][][][];

			for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
				for (int u2 = 2; u2 < u1; u2++) {					
					int pos1 = u1*u1val + u2;
					probs_[u1][pos1] = new double[paramsPerAtt[u2] * u2][][];

					for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
						for (int u3 = 1; u3 < u2; u3++) {
							int pos2 = u2*u2val + u3;
							probs_[u1][pos1][pos2] = new double[paramsPerAtt[u3] * u3][];

							for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {
								for (int u4 = 0; u4 < u3; u4++) {
									int pos3 = u3*u3val + u4;
									probs_[u1][pos1][pos2][pos3] = new double[paramsPerAtt[u4] * nc];
								}
							}
						}
					}
				}
			}
		}

		for (int c = 0; c < nc; c++) {

			for (int u1 = 3; u1 < n; u1++) {
				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

					for (int u2 = 2; u2 < u1; u2++) {
						for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

							for (int u3 = 1; u3 < u2; u3++) {
								for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

									for (int u4 = 0; u4 < u3; u4++) {
										for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

											int pos1 = u1*u1val + u2;
											int pos2 = u2*u2val + u3;
											int pos3 = u3*u3val + u4;
											int pos4 = u4val*nc + c;	

											probs_[u1][pos1][pos2][pos3][pos4] = Math.max(SUtils.MEsti(ref(u1,u1val,u2,u2val,u3,u3val,u4,u4val,c), 
													xxxyDist_.xxyDist_.xyDist_.getClassCount(c), 
													paramsPerAtt[u1]*paramsPerAtt[u2]*paramsPerAtt[u3]*paramsPerAtt[u4]), 1e-75);
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

	// p(x1=v1, x2=v2, x3=v3, x4=v4, Y=y) unsmoothed
	public double rawJointP(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4, int y) {
		return ref(x1, v1, x2, v2, x3, v3, x4, v4, y) / N;
	}

	// p(x1=v1, x2=v2, x3=v3,x4=v4, Y=y) using M-estimate
	public double jointP(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4, int y) {
		//return (*constRef(x1, v1, x2, v2, x3, v3, x4, v4, y) + M / (instanceStream_->getNoValues(x1) * instanceStream_->getNoValues(x2) * instanceStream_->getNoValues(x3) * noClasses_)) / (xxxyCounts.xxyCounts.xyCounts.count + M);
		return SUtils.MEsti(ref(x1, v1, x2, v2, x3, v3, x4, v4, y), N, paramsPerAtt[x1] * paramsPerAtt[x2] * paramsPerAtt[x3] * paramsPerAtt[x4] * nc);
	}

	// p(x1=v1, x2=v2, x3=v3,x4=v4) using M-estimate
	public double jointP(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4) {
		//return (*constRef(x1, v1, x2, v2, x3, v3, x4, v4, y) + M / (instanceStream_->getNoValues(x1) * instanceStream_->getNoValues(x2) * instanceStream_->getNoValues(x3) * noClasses_)) / (xxxyCounts.xxyCounts.xyCounts.count + M);
		return SUtils.MEsti(getCount(x1, v1, x2, v2, x3, v3, x4, v4), N, paramsPerAtt[x1] * paramsPerAtt[x2] * paramsPerAtt[x3] * paramsPerAtt[x4]);
	}

	// p(x1=v1|Y=y, x2=v2, x3=v3,x4=v4) using M-estimate
	public double p(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4, int y) {
		//return (*constRef(x1, v1, x2, v2, x3, v3, x4, v4, y) + M / instanceStream_->getNoValues(x1)) / (xxxyCounts.getCount(x2, v2, x3, v3, x4, v4, y) + M);
		return SUtils.MEsti(ref(x1, v1, x2, v2, x3, v3, x4, v4, y), xxxyDist_.getCount(x2, v2, x3, v3, x4, v4, y), paramsPerAtt[x1]);
	}
	
	// p(x1=v1, x2=v2, x3=v3,x4=v4|Y=y) using M-estimate
	public double jp(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4, int y) {
		return jref(x1, v1, x2, v2, x3, v3, x4, v4, y);
	}

	// get count for instance (x1=v1, x2=v2, x3=v3,x4=v4, Y=y)
	public double getCount(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4, int y) {
		return ref(x1, v1, x2, v2, x3, v3, x4, v4, y);
	}

	// count for instances x1=v1, x2=v2,x3=v3
	public int getCount(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4) {
		int c = 0;

		for (int y = 0; y < nc; y++) {
			c += getCount(x1, v1, x2, v2, x3, v3, x4, v4, y);
		}
		return c;
	}

	// count[X1=x1][X2=x2][X3=x3][X4=x4][Y=y]
	private double ref(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4, int y) {

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
		if (x4 > x3) {
			int t = x3;
			x3 = x4;
			x4 = t;
			t = v3;
			v3 = v4;
			v4 = t;
		}

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
		assert(x1 > x2 && x2 > x3 && x3 > x4);
		//return &count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * x3 + x4][v4 * noClasses_ + y];

		int pos1 = v1*x1 + x2;
		int pos2 = v2*x2 + x3;
		int pos3 = v3*x3 + x4;
		int pos4 = v4*nc + y;

		return counts_[x1][pos1][pos2][pos3][pos4];
	}
	
	// probs[X1=x1][X2=x2][X3=x3][X4=x4][Y=y]
	private double jref(int x1, int v1, int x2, int v2, int x3, int v3, int x4, int v4, int y) {

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
		if (x4 > x3) {
			int t = x3;
			x3 = x4;
			x4 = t;
			t = v3;
			v3 = v4;
			v4 = t;
		}

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
		assert(x1 > x2 && x2 > x3 && x3 > x4);
		//return &count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * x3 + x4][v4 * noClasses_ + y];

		int pos1 = v1*x1 + x2;
		int pos2 = v2*x2 + x3;
		int pos3 = v3*x3 + x4;
		int pos4 = v4*nc + y;

		return probs_[x1][pos1][pos2][pos3][pos4];
	}

	public int getNoAtts() { return n; }

	public int getNoCatAtts() { return n; }

	public int getNoValues(int a) { return paramsPerAtt[a]; }

	public int getNoData() { return N; }

	public int getNoClasses() { return nc; }

	public int[] getNoValues() { return paramsPerAtt; }

}

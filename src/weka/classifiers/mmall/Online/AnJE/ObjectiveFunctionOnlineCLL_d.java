package weka.classifiers.mmall.Online.AnJE;

import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParameters;
import weka.classifiers.mmall.Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class ObjectiveFunctionOnlineCLL_d extends ObjectiveFunctionOnlineCLL {

	public ObjectiveFunctionOnlineCLL_d(wdAnJEOnline algorithm) {
		super(algorithm);		
	}
	
	@Override
	public void update(Instance instance, int t, double[] results) {
		
		//algorithm.printParameters();
		
		double negLogLikelihood = 0.0;
		double error = 0;
		double RMSE = 0;
		
		String m_S = algorithm.getMS();
		algorithm.dParameters_.resetGradients();

		int n = algorithm.getnAttributes();
		int nc = algorithm.getNc();

		boolean m_Regularization = algorithm.getRegularization();
		double m_Eta = algorithm.getEta();
		double m_Lambda = algorithm.getLambda();		

		double[] myProbs = new double[nc];

		wdAnJEParameters dParameters = algorithm.getdParameters_();

		double mLogNC = -Math.log(nc);		

		// ------------------------------------
		// Update Gradient
		// ------------------------------------
		int x_C = (int) instance.classValue();
		myProbs = algorithm.logDistributionForInstance(instance);
		negLogLikelihood += (mLogNC - myProbs[x_C]);
		SUtils.exp(myProbs);	
		
		// ------------------------------------
		// Update Error and RMSE
		// ------------------------------------
		int pred = -1;
		double bestProb = Double.MIN_VALUE;
		for (int y = 0; y < nc; y++) {
			if (!Double.isNaN(myProbs[y])) {
				if (myProbs[y] > bestProb) {
					pred = y;
					bestProb = myProbs[y];
				}
				RMSE += (1/(double)nc) * Math.pow((myProbs[y]-((y==x_C)?1:0)), 2);

			} else {
				System.err.println("probs[ " + y + "] is NaN! oh no!");
			}
		}

		if (pred != x_C) {
			error = 1;
		}
		
		results[0] += negLogLikelihood;
		results[1] += error;
		results[2] += RMSE;

		//algorithm.logGradientForInstance_d(g, myProbs, instance);
		// -------------------------------------------------------			
		for (int c = 0; c < nc; c++) {			
			if (m_Regularization)
				dParameters.setGradientAtFullIndex(c, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * dParameters.getParameterAtFullIndex(c));
			else
				dParameters.setGradientAtFullIndex(c, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
		}

		if (m_S.equalsIgnoreCase("A1JE")) {
			// A1JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 0; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					long index = dParameters.getAttributeIndex(att1, att1val, c);
					if (m_Regularization)
						dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * dParameters.getParameterAtFullIndex(index));
					else
						dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
				}
			}

		} else if (m_S.equalsIgnoreCase("A2JE")) {
			// A2JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 1; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int att2 = 0; att2 < att1; att2++) {
						int att2val = (int) instance.value(att2);

						long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, c);
						if (m_Regularization)
							dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * dParameters.getParameterAtFullIndex(index));
						else
							dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
					}
				}
			}

		} else if (m_S.equalsIgnoreCase("A3JE")) {
			// A3JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 2; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int att2 = 1; att2 < att1; att2++) {
						int att2val = (int) instance.value(att2);

						for (int att3 = 0; att3 < att2; att3++) {
							int att3val = (int) instance.value(att3);

							long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
							if (m_Regularization)
								dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * dParameters.getParameterAtFullIndex(index));
							else
								dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
						}
					}
				}
			}

		} else if (m_S.equalsIgnoreCase("A4JE")) {
			// A4JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 3; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int att2 = 2; att2 < att1; att2++) {
						int att2val = (int) instance.value(att2);

						for (int att3 = 1; att3 < att2; att3++) {
							int att3val = (int) instance.value(att3);

							for (int att4 = 0; att4 < att3; att4++) {
								int att4val = (int) instance.value(att4);

								long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, c);
								if (m_Regularization)
									dParameters.setGradientAtFullIndex(index,(-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * dParameters.getParameterAtFullIndex(index));
								else
									dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
							}
						}
					}
				}
			}

		} else if (m_S.equalsIgnoreCase("A5JE")) {
			// A5JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 4; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int att2 = 3; att2 < att1; att2++) {
						int att2val = (int) instance.value(att2);

						for (int att3 = 2; att3 < att2; att3++) {
							int att3val = (int) instance.value(att3);

							for (int att4 = 1; att4 < att3; att4++) {
								int att4val = (int) instance.value(att4);

								for (int att5 = 0; att5 < att4; att5++) {
									int att5val = (int) instance.value(att5);

									long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, att5, att5val, c);
									if (m_Regularization)
										dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) + m_Lambda * dParameters.getParameterAtFullIndex(index));
									else
										dParameters.setGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));

								}
							}
						}
					}
				}
			}

		} else {
			System.out.println("m_S value should be from set {A1JE, A2JE, A3JE, A4JE, A5JE}");
		}

		// -------------------------------------------------------	

		// ------------------------------------
		// Update Parameters
		// ------------------------------------
		double factor = 0;
		if (m_Regularization) {
			factor = m_Eta / (1 + m_Lambda * t);
		} else {
			factor = m_Eta / (1 + t);
		}

		for (int c = 0; c < nc; c++) {		
			dParameters.setParameterAtFullIndex(c, dParameters.getParameterAtFullIndex(c) - factor * dParameters.getGradientAtFullIndex(c));
		}
		if (m_S.equalsIgnoreCase("A1JE")) {
			// A1JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 0; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					long index = dParameters.getAttributeIndex(att1, att1val, c);
					dParameters.setParameterAtFullIndex(index, dParameters.getParameterAtFullIndex(index) - factor * dParameters.getGradientAtFullIndex(index));
				}
			}

		} else if (m_S.equalsIgnoreCase("A2JE")) {
			// A2JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 1; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int att2 = 0; att2 < att1; att2++) {
						int att2val = (int) instance.value(att2);

						long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, c);
						dParameters.setParameterAtFullIndex(index, dParameters.getParameterAtFullIndex(index) - factor * dParameters.getGradientAtFullIndex(index));
					}
				}				
			}

		} else if (m_S.equalsIgnoreCase("A3JE")) {
			// A3JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 2; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int att2 = 1; att2 < att1; att2++) {
						int att2val = (int) instance.value(att2);

						for (int att3 = 0; att3 < att2; att3++) {
							int att3val = (int) instance.value(att3);

							long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);							
							
							dParameters.setParameterAtFullIndex(index, dParameters.getParameterAtFullIndex(index) - factor * dParameters.getGradientAtFullIndex(index));
							
							//double value = dParameters.getParameterAtFullIndex(index) - factor * dParameters.getGradientAtFullIndex(index);							
							//System.out.println(c + ", " + att1 + ":" + att1val + ", " + att2 + ":" + att2val + ", " + att3 + ":" + att3val + ", " + value + ", " + dParameters.getParameterAtFullIndex(index) + ", " + factor * dParameters.getGradientAtFullIndex(index));
							//dParameters.setParameterAtFullIndex(index, value);
						}
					}
				}
			}

		} else if (m_S.equalsIgnoreCase("A4JE")) {
			// A4JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 3; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int att2 = 2; att2 < att1; att2++) {
						int att2val = (int) instance.value(att2);

						for (int att3 = 1; att3 < att2; att3++) {
							int att3val = (int) instance.value(att3);

							for (int att4 = 0; att4 < att3; att4++) {
								int att4val = (int) instance.value(att4);

								long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, c);
								dParameters.setParameterAtFullIndex(index, dParameters.getParameterAtFullIndex(index) - factor * dParameters.getGradientAtFullIndex(index));
							}
						}
					}
				}
			}

		} else if (m_S.equalsIgnoreCase("A5JE")) {
			// A5JE

			for (int c = 0; c < nc; c++) {
				for (int att1 = 3; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int att2 = 2; att2 < att1; att2++) {
						int att2val = (int) instance.value(att2);

						for (int att3 = 1; att3 < att2; att3++) {
							int att3val = (int) instance.value(att3);

							for (int att4 = 0; att4 < att3; att4++) {
								int att4val = (int) instance.value(att4);

								for (int att5 = 0; att5 < att4; att5++) {
									int att5val = (int) instance.value(att5);

									long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, att5, att5val, c);
									dParameters.setParameterAtFullIndex(index, dParameters.getParameterAtFullIndex(index) - factor * dParameters.getGradientAtFullIndex(index));
								}
							}
						}
					}
				}
			}

		} else {
			System.out.println("m_S value should be from set {A1JE, A2JE, A3JE, A4JE, A5JE}");
		}
		
//		double magGrad = 0;
////		magGrad = SUtils.maxAbsValueInAnArray(g);
//		
//		for (int i = 0; i < g.length; i++) {
//			magGrad += g[i] * g[i];
//		}
//		magGrad = Math.sqrt(magGrad);
//		
	}

	@Override
	public void update(Instances m_Instances) {
		// TODO Auto-generated method stub
		
	}
	
}



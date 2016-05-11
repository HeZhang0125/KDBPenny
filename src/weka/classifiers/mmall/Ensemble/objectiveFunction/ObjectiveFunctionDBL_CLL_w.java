package weka.classifiers.mmall.Ensemble.objectiveFunction;

import lbfgsb.FunctionValues;
import weka.classifiers.mmall.DataStructure.DBL.DBLParameters;
import weka.classifiers.mmall.Ensemble.DBL;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class ObjectiveFunctionDBL_CLL_w extends ObjectiveFunctionDBL {

	public ObjectiveFunctionDBL_CLL_w(DBL algorithm) {
		super(algorithm);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		double negLogLikelihood = 0.0;
		String m_S = algorithm.getMS();

		algorithm.dblParameters_.copyParameters(params);
		algorithm.dblParameters_.resetGradients();

		int n = algorithm.getnAttributes();
		int nc = algorithm.getNc();

		double[] myProbs = new double[nc];

		DBLParameters dblParameters_ = algorithm.getdParameters_();
		Instances instances = algorithm.getM_Instances();
		int N = instances.numInstances();

		double mLogNC = -Math.log(nc); 

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);
			int x_C = (int) instance.classValue();
			algorithm.logDistributionForInstance(myProbs,instance);
			negLogLikelihood += (mLogNC - myProbs[x_C]);
			SUtils.exp(myProbs);

			//algorithm.logGradientForInstance_w(g, myProbs, instance);
			// -------------------------------------------------------
			for (int c = 0; c < nc; c++) {
				dblParameters_.incGradientAtFullIndex(c, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(c));
			}

			if (m_S.equalsIgnoreCase("A1JE")) {
				// A1JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 0; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						long index = dblParameters_.getAttributeIndex(att1, att1val, c);
						dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));
					}
				}

			} else if (m_S.equalsIgnoreCase("A2JE")) {
				// A2JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 0; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						long index = dblParameters_.getAttributeIndex(att1, att1val, c);
						dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

						for (int att2 = 0; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, c);
							dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A3JE")) {
				// A3JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 0; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						long index = dblParameters_.getAttributeIndex(att1, att1val, c);
						dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

						for (int att2 = 0; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, c);
							dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

							for (int att3 = 0; att3 < att2; att3++) {
								int att3val = (int) instance.value(att3);

								index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
								dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));
							}
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A4JE")) {
				// A4JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 0; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						long index = dblParameters_.getAttributeIndex(att1, att1val, c);
						dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

						for (int att2 = 0; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, c);
							dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

							for (int att3 = 0; att3 < att2; att3++) {
								int att3val = (int) instance.value(att3);

								index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
								dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

								for (int att4 = 0; att4 < att3; att4++) {
									int att4val = (int) instance.value(att4);

									index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, c);
									dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));
								}
							}
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A5JE")) {
				// A5JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 0; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						long index = dblParameters_.getAttributeIndex(att1, att1val, c);
						dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

						for (int att2 = 0; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, c);
							dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

							for (int att3 = 0; att3 < att2; att3++) {
								int att3val = (int) instance.value(att3);

								index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
								dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

								for (int att4 = 0; att4 < att3; att4++) {
									int att4val = (int) instance.value(att4);

									index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, c);
									dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));

									for (int att5 = 0; att5 < att4; att5++) {
										int att5val = (int) instance.value(att5);

										index = dblParameters_.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, att5, att5val, c);
										dblParameters_.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dblParameters_.getProbAtFullIndex(index));
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
		}

//		if (algorithm.isM_MVerb()) {
//			System.out.print(negLogLikelihood + ", ");			
//		}

		return new FunctionValues(negLogLikelihood, dblParameters_.getGradients());
	}

}

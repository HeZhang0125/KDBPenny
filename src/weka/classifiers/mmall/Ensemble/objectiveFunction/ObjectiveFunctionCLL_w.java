package weka.classifiers.mmall.Ensemble.objectiveFunction;

import lbfgsb.FunctionValues;
import weka.classifiers.mmall.Ensemble.wdAnJE;
import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParameters;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class ObjectiveFunctionCLL_w extends ObjectiveFunction {

	public ObjectiveFunctionCLL_w(wdAnJE algorithm) {
		super(algorithm);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		double negLogLikelihood = 0.0;
		String m_S = algorithm.getMS();

		algorithm.dParameters_.copyParameters(params);
		algorithm.dParameters_.resetGradients();

		int n = algorithm.getnAttributes();
		int nc = algorithm.getNc();

		double[] myProbs = new double[nc];

		wdAnJEParameters dParameters = algorithm.getdParameters_();
		Instances instances = algorithm.getM_Instances();
		int N = instances.numInstances();

		boolean m_Regularization = algorithm.getRegularization();
		double m_Lambda = algorithm.getLambda();

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
				if (m_Regularization) {
					negLogLikelihood += m_Lambda/2 * dParameters.getParameterAtFullIndex(c) * dParameters.getParameterAtFullIndex(c);
					dParameters.incGradientAtFullIndex(c, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(c) + m_Lambda * dParameters.getParameterAtFullIndex(c));
				} else {
					dParameters.incGradientAtFullIndex(c, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(c));
				}
			}

			if (m_S.equalsIgnoreCase("A1JE")) {
				// A1JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 0; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						long index = dParameters.getAttributeIndex(att1, att1val, c);
						if (m_Regularization) {
							negLogLikelihood += m_Lambda/2 * dParameters.getParameterAtFullIndex(index) * dParameters.getParameterAtFullIndex(index);
							dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index) + m_Lambda * dParameters.getParameterAtFullIndex(index));
						} else {							
							dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index));
						}
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
							if (m_Regularization) {
								negLogLikelihood += m_Lambda/2 * dParameters.getParameterAtFullIndex(index) * dParameters.getParameterAtFullIndex(index);
								dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index) + m_Lambda * dParameters.getParameterAtFullIndex(index));
							} else {															
								dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index));
							}
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

								if (m_Regularization) {
									negLogLikelihood += m_Lambda/2 * dParameters.getParameterAtFullIndex(index) * dParameters.getParameterAtFullIndex(index);
									dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index) + m_Lambda * dParameters.getParameterAtFullIndex(index));
								} else {																
									dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index));
								}
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

									if (m_Regularization) {
										negLogLikelihood += m_Lambda/2 * dParameters.getParameterAtFullIndex(index) * dParameters.getParameterAtFullIndex(index);
										dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index) + m_Lambda * dParameters.getParameterAtFullIndex(index));
									} else {																	
										dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index));
									}
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

										if (m_Regularization) {
											negLogLikelihood += m_Lambda/2 * dParameters.getParameterAtFullIndex(index) * dParameters.getParameterAtFullIndex(index);
											dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index) + m_Lambda * dParameters.getParameterAtFullIndex(index));
										} else {											
											dParameters.incGradientAtFullIndex(index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getProbAtFullIndex(index));
										}
									}
								}
							}
						}
					}
				}

			} else {
				System.out.println("m_S value should be from set {A1JE, A2JE, A3JE, A4JE}");
			}
			// -------------------------------------------------------
			
		}

		return new FunctionValues(negLogLikelihood, dParameters.getGradients());
	}

}

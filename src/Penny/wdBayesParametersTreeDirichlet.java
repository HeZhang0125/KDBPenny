package Penny;

import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
import weka.core.Instances;

public class wdBayesParametersTreeDirichlet extends wdBayesParametersTree {

	public wdBayesParametersTreeDirichlet(Instances instances, int[] paramsPerAtt, int[] m_Order, int[][] m_Parents,
			int m_P) {
		super(instances, paramsPerAtt, m_Order, m_Parents, m_P);
	}

	public void countsToProbability(xxyDist xxyDist_) {

		for (int u = 0; u < n; u++) {
			// convert all probabilities into Laplace estimation first
			super.convertCountToProbs(order[u], parents[u], wdBayesNode_[u]);

			probU = new double[m_ParamsPerAtt[u]]; // P(u)
			for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
				probU[uval] = xxyDist_.xyDist_.p(u, uval);// P(uval)
			}

			probY = new double[nc];
			for (int y = 0; y < nc; y++) {
				probY[y] = xxyDist_.xyDist_.pp(y);// P(y)
			}

			// p^(u|y) = w1*p(u|y)+ w2*p(u)
			for (int y = 0; y < nc; y++) {
				double w1 = computeW1(probY[y]);
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					double p = wdBayesNode_[u].getXYProbability(uval, y);
					double prob = smooth(w1, p, probU[uval]);
					wdBayesNode_[u].setXYProbability(uval, y, prob);
				}
			}

			convertCountToProbs(order[u], wdBayesNode_[u], nc);
		}
	}

	public void convertCountToProbs(int u, wdBayesNode pt, int numParentValues) {

		int att = pt.att;
		if (att != -1) {
			int[][] tempArray = new int[nc][m_ParamsPerAtt[u]];
			numParentValues *= m_ParamsPerAtt[att];
			int numChildren = pt.children.length;

			for (int j = 0; j < numChildren; j++) {

				if (pt.children[j] != null) {
					wdBayesNode next = pt.children[j];

					for (int y = 0; y < nc; y++) {
						int a = 0;
						for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
							tempArray[y][uval] = (int) next.getXYCount(uval, y);
							a += tempArray[y][uval];
						}
						double pp = KDBPenny.laplace(a, nInstances, numParentValues);
						double w1 = computeW1(pp);
						if (pp > 1) {
							System.out.println(a + "  " + nInstances + "  " + numParentValues);
						}
						for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {

							double p = next.getXYProbability(uval, y);
							double prob = smooth(w1, p, pt.getXYProbability(uval, y));
							next.setXYProbability(uval, y, prob);
						}
					}
					convertCountToProbs(u, next, numParentValues);
				}
				// Flag end of nodes
				att = -1;
			}
		}
		return;
	}
}

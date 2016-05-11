package Penny;

import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
import weka.core.Instances;

public class wdBayesParametersTreeLaplace extends wdBayesParametersTree {

	public wdBayesParametersTreeLaplace(Instances instances, int[] paramsPerAtt, int[] m_Order, int[][] m_Parents,
			int m_P) {
		super(instances, paramsPerAtt, m_Order, m_Parents, m_P);
	}

	public void convertCountToProbs(int u, int[] lparents, wdBayesNode pt) {
		int att = pt.att;

		if (att == -1) {// leaf node
			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
				}
			}
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					int denom = 0;
					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
						denom += tempArray[dval][y];
					}
					double prob1 = KDBPenny.laplace(tempArray[uval][y], denom, m_ParamsPerAtt[u]);
					pt.setXYProbability(uval, y, prob1);
				}
			}
			return;
		}

		while (att != -1) {
			/*
			 * Now convert non-leaf node counts to probs
			 */
			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
				}
			}
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					int denom = 0;
					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
						denom += tempArray[dval][y];
					}

					double prob = KDBPenny.laplace(tempArray[uval][y], denom, m_ParamsPerAtt[u]);
					pt.setXYProbability(uval, y, prob);
				}
			}

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					convertCountToProbs(u, lparents, next);
				// Flag end of nodes
				att = -1;
			}
		}
		return;
	}
}

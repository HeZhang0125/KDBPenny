package weka.classifiers.mmall.Evaluation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

public class ERMSEminimizer_V3 {

    public static int[] x1D = { 0, 1 };
    public static int[] x2D = { 0, 1 };
    public static int[] yD = { 0, 1 };

    public static void main(String[] args) throws Exception {
	int seed = 3071980;
	int nInstances = 1000;

	double[][][] pj = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };

	double[][][] py_x1x2 = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };
	double[][] p_x1x2 = { { 0.0, 0.0 }, { 0.0, 0.0 } };

	double[] c = { 0.75, 0.25 };
	// double[] c = {0.01,0.99};
	double a1, a2, a3, a4;
	double l1, l2, l3, l4, l5, l6, l7, l8;

	// Francois' variables
	double[] pjTab = new double[8];
	RandomGenerator rg = new MersenneTwister();
	rg.setSeed(seed);

	int counter = 0;
	for (int i1 = 0; i1 < c.length; i1++) {
	    for (int i2 = 0; i2 < c.length; i2++) {
		for (int i3 = 0; i3 < c.length; i3++) {
		    for (int i4 = 0; i4 < c.length; i4++) {

			a1 = c[i1];
			a2 = c[i2];
			a3 = c[i3];
			a4 = c[i4];

			py_x1x2[0][0][0] = a1; // 0.25; //a1;
			py_x1x2[0][0][1] = a2; // 0.25; //a2;
			py_x1x2[0][1][0] = a3; // 0.25; //a3;
			py_x1x2[0][1][1] = a4; // 0.25; //a4;
			py_x1x2[1][0][0] = 1 - a1; // 0.75; //1 - a1;
			py_x1x2[1][0][1] = 1 - a2; // 0.75; //1 - a2;
			py_x1x2[1][1][0] = 1 - a3; // 0.75; //1 - a3;
			py_x1x2[1][1][1] = 1 - a4; // 0.75; //1 - a4;

			for (double j = 0; j <= 0.5; j += 0.1) {
			    // p_x1x2[0][0] = j;
			    // p_x1x2[0][1] = 0.5 - j;
			    // p_x1x2[1][0] = 0.5 - j;
			    // p_x1x2[1][1] = j;
			    p_x1x2[0][0] = 0.5 - j;
			    p_x1x2[0][1] = j;
			    p_x1x2[1][0] = j;
			    p_x1x2[1][1] = 0.5 - j;

			    for (int y = 0; y < yD.length; y++) {
				for (int x1 = 0; x1 < x1D.length; x1++) {
				    for (int x2 = 0; x2 < x2D.length; x2++) {
					pj[y][x1][x2] = p_x1x2[x1][x2] * py_x1x2[y][x1][x2];
				    }
				}
			    }

			    l1 = pj[0][0][0];
			    l2 = pj[0][0][1];
			    l3 = pj[0][1][0];
			    l4 = pj[0][1][1];
			    l5 = pj[1][0][0];
			    l6 = pj[1][0][1];
			    l7 = pj[1][1][0];
			    l8 = pj[1][1][1];

			    System.out.printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", l1, l2, l3, l4, l5, l6, l7, l8);

			    double CMI = computeCMP(pj);
			    File arffFile = new File("/tmp/data" + "_" + counter + "_" + CMI);

			    // Francois' code to sample joint ===========
			    PrintWriter arff = new PrintWriter(new BufferedWriter(new FileWriter(arffFile), 1000000));
			    arff.println("@relation cmi\n");
			    arff.println("@attribute x1 {0,1}");
			    arff.println("@attribute x2 {0,1}");
			    arff.println("@attribute y {0,1}");
			    arff.println();
			    arff.println("@data");

			    // linearizing pj and computing cfd
			    pjTab[0] = pj[0][0][0];
			    pjTab[1] = pj[0][0][1];
			    pjTab[2] = pj[0][1][0];
			    pjTab[3] = pj[0][1][1];
			    pjTab[4] = pj[1][0][0];
			    pjTab[5] = pj[1][0][1];
			    pjTab[6] = pj[1][1][0];
			    pjTab[7] = pj[1][1][1];

			    for (int i = 0; i < nInstances; i++) {
				double rand = rg.nextDouble();
				int chosenValue = 0;
				double sumProba = pjTab[chosenValue];
				while (rand > sumProba) {
				    chosenValue++;
				    if (chosenValue >= pjTab.length) {
					System.err.println("pjTab does not sum up to 1!");
					System.err.println(Arrays.toString(pjTab));
					System.err.println(sumProba);
					System.err.println(rand);
				    }
				    sumProba += pjTab[chosenValue];
				}
				int x2Val = chosenValue & 1;
				chosenValue = chosenValue >>> 1;
				int x1Val = chosenValue & 1;
				chosenValue = chosenValue >>> 1;
				int yVal = chosenValue & 1;
//				System.out.println( " -> (" + yVal + "," + x1Val + "," + x2Val + ")");
				arff.println(x1Val+","+x2Val+","+yVal);

			    }
			    arff.flush();
			    arff.close();

			    // End of Francois' code to sample joint ====

			    counter++;
			}

			System.out.println("All Done");
		    }
		}
	    }
	}

    }

    public static double computeCMP(double[][][] pj) {

	/*
	 * Compute prior probability p(y)
	 */
	double[] py = { 0, 0 };
	for (int y = 0; y < yD.length; y++) {
	    py[y] = (pj[y][0][0] + pj[y][0][1]) + (pj[y][1][0] + pj[y][1][1]);
	}

	/*
	 * Compute conditional probability of each attribute read as p(x1|y) and
	 * p(x2|y)
	 */
	double[][][] px1_y = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };
	double[][][] px2_y = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };
	for (int y = 0; y < yD.length; y++) {
	    for (int x1 = 0; x1 < x1D.length; x1++) {
		for (int x2 = 0; x2 < x2D.length; x2++) {
		    if (py[y] == 0) {
			px1_y[y][x1][x2] = 0;
			px2_y[y][x1][x2] = 0;
		    } else {
			px1_y[y][x1][x2] = (pj[y][x1][0] + pj[y][x1][1]) / py[y];
			px2_y[y][x1][x2] = (pj[y][0][x2] + pj[y][1][x2]) / py[y];
		    }
		}
	    }
	}

	/*
	 * Compute conditional mutual information formula: \sum_x1,x2,y
	 * p(x1,x2,y) log( p(x1,x2|y)/ p(x1|y)p(x2|y) )
	 */
	double CMI = 0.0;
	for (int y = 0; y < yD.length; y++) {
	    for (int x1 = 0; x1 < x1D.length; x1++) {
		for (int x2 = 0; x2 < x2D.length; x2++) {
		    double CMI_denom = px1_y[y][x1][x2] * px2_y[y][x1][x2];
		    if (CMI_denom == 0 || pj[y][x1][x2] == 0)
			CMI = CMI + 0;
		    else
			CMI = CMI + pj[y][x1][x2] * Math.log((pj[y][x1][x2] / py[y]) / (px1_y[y][x1][x2] * px2_y[y][x1][x2]));
		}
	    }
	}

	return CMI;
    }

    public static void computeExpError(double[][][] pj) {

	/*
	 * Compute prior probability p(y)
	 */
	double[] py = { 0, 0 };
	for (int y = 0; y < yD.length; y++) {
	    py[y] = (pj[y][0][0] + pj[y][0][1]) + (pj[y][1][0] + pj[y][1][1]);
	}

	/*
	 * Compute conditional probability of each attribute read as p(x1|y) and
	 * p(x2|y)
	 */
	double[][][] px1_y = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };
	double[][][] px2_y = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };
	for (int y = 0; y < yD.length; y++) {
	    for (int x1 = 0; x1 < x1D.length; x1++) {
		for (int x2 = 0; x2 < x2D.length; x2++) {
		    if (py[y] == 0) {
			px1_y[y][x1][x2] = 0;
			px2_y[y][x1][x2] = 0;
		    } else {
			px1_y[y][x1][x2] = (pj[y][x1][0] + pj[y][x1][1]) / py[y];
			px2_y[y][x1][x2] = (pj[y][0][x2] + pj[y][1][x2]) / py[y];
		    }
		}
	    }
	}

	/*
	 * Compute the conditional probability read as p(y|x1,x2) e.g.,
	 * p(y=0|x1=1,x2=1)
	 */
	double[][][] py_x1x2 = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };
	for (int y = 0; y < yD.length; y++) {
	    for (int x1 = 0; x1 < x1D.length; x1++) {
		for (int x2 = 0; x2 < x2D.length; x2++) {
		    double denom = (pj[0][x1][x2] + pj[1][x1][x2]);
		    if (denom == 0)
			py_x1x2[y][x1][x2] = 0;
		    else
			py_x1x2[y][x1][x2] = pj[y][x1][x2] / denom;
		}
	    }
	}

	/*
	 * Compute NB error NB(y|x1,x2) = p(y)p(x1|y)p(x2|y) where p(x2|y) =
	 * (p(x1=0|x2,y) + p(x1=1|x2,y))/p(y)
	 * 
	 * Note: This is not going to be used, only for checking, NB error will
	 * be computed in WANB block when w1 = 1 and w2 = 1
	 */
	double[][][] nb = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };
	double NBerrorSq = 0.0;
	for (int y = 0; y < yD.length; y++) {
	    for (int x1 = 0; x1 < x1D.length; x1++) {
		for (int x2 = 0; x2 < x2D.length; x2++) {
		    double nb_denom = ((py[0] * px1_y[0][x1][x2] * px2_y[0][x1][x2]) + (py[1] * px1_y[1][x1][x2] * px2_y[1][x1][x2]));
		    if (nb_denom == 0)
			nb[y][x1][x2] = 0;
		    else
			nb[y][x1][x2] = (py[y] * px1_y[y][x1][x2] * px2_y[y][x1][x2]) / nb_denom;

		    NBerrorSq = NBerrorSq + pj[y][x1][x2] * Math.pow(py_x1x2[y][x1][x2] - nb[y][x1][x2], 2);
		}
	    }
	}

	/*
	 * Compute conditional mutual information formula: \sum_x1,x2,y
	 * p(x1,x2,y) log( p(x1,x2|y)/ p(x1|y)p(x2|y) )
	 */
	double CMI = 0.0;
	for (int y = 0; y < yD.length; y++) {
	    for (int x1 = 0; x1 < x1D.length; x1++) {
		for (int x2 = 0; x2 < x2D.length; x2++) {
		    double CMI_denom = px1_y[y][x1][x2] * px2_y[y][x1][x2];
		    if (CMI_denom == 0 || pj[y][x1][x2] == 0)
			CMI = CMI + 0;
		    else
			CMI = CMI + pj[y][x1][x2] * Math.log((pj[y][x1][x2] / py[y]) / (px1_y[y][x1][x2] * px2_y[y][x1][x2]));
		}
	    }
	}

	/* Compute p(x_1) */
	double[] px1 = { 0, 0 };
	for (int x1 = 0; x1 < x1D.length; x1++) {
	    px1[x1] = (pj[0][x1][0] + pj[0][x1][1]) + (pj[1][x1][0] + pj[1][x1][1]);
	}

	/* Compute p(x_2) */
	double[] px2 = { 0, 0 };
	for (int x2 = 0; x2 < x2D.length; x2++) {
	    px2[x2] = (pj[0][0][x2] + pj[0][1][x2]) + (pj[1][0][x2] + pj[1][1][x2]);
	}

	/*
	 * Compute mutual information formula: \sum_x1,x2,y p(x1,x2,y) log(
	 * p(x1,x2|y)/ p(x1|y)p(x2|y) )
	 */
	double MI = 0.0;
	for (int x1 = 0; x1 < x1D.length; x1++) {
	    for (int x2 = 0; x2 < x2D.length; x2++) {
		double MI_denom = (px1[x1] * px2[x2]);
		if (MI_denom == 0 || pj[0][x1][x2] == 0 || pj[1][x1][x2] == 0)
		    MI = MI + 0;
		else
		    MI = MI + (pj[0][x1][x2] + pj[1][x1][x2]) * Math.log((pj[0][x1][x2] + pj[1][x1][x2]) / (px1[x1] * px2[x2]));
	    }
	}

	// double MI = 0.0;
	// for (int y = 0; y < yD.length; y++) {
	// for (int x1 = 0; x1 < x1D.length; x1++) {
	// for (int x2 = 0; x2 < x2D.length; x2++) {
	// //double MI_denom = px1_y[y][x1][x2] * px2_y[y][x1][x2];
	// //if (MI_denom == 0 || pj[y][x1][x2] == 0)
	// // MI = MI + 0;
	// //else
	// MI = MI + pj[y][x1][x2] * Math.pow( (pj[y][x1][x2] / py[y]) -
	// (px1_y[y][x1][x2] * px2_y[y][x1][x2]), 2);
	// }
	// }
	// }

	/* Compute error for Weighted NB */
	double[] w1 = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
	double[] w2 = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
	double minError = Double.MAX_VALUE;
	double minErrorMatrix[][] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
	double opt_w1 = 0, opt_w2 = 0, errorNone = 0, errorW1 = 0, errorW2 = 0, errorW1W2 = 0;

	for (int i = 0; i < w1.length; i++) {
	    for (int j = 0; j < w2.length; j++) {

		// WANB(y|x1,x2) = p(y) * p(x1|y)^w1 * p(x2|y)^w2
		// error^2(y,x1,x2) = p(y|x1,x2) - WANB(y|x1,x2)
		double[][][] wanb = { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } };
		double error = 0.0;
		for (int y = 0; y < yD.length; y++) {
		    for (int x1 = 0; x1 < x1D.length; x1++) {
			for (int x2 = 0; x2 < x2D.length; x2++) {
			    double wanb_denom = (py[0] * Math.pow(px1_y[0][x1][x2], w1[i]) * Math.pow(px2_y[0][x1][x2], w2[j]) + py[1]
				    * Math.pow(px1_y[1][x1][x2], w1[i]) * Math.pow(px2_y[1][x1][x2], w2[j]));

			    if (wanb_denom == 0)
				wanb[y][x1][x2] = 0;
			    else
				wanb[y][x1][x2] = py[y] * Math.pow(px1_y[y][x1][x2], w1[i]) * Math.pow(px2_y[y][x1][x2], w2[j]) / wanb_denom;

			    error = error + pj[y][x1][x2] * Math.pow(py_x1x2[y][x1][x2] - wanb[y][x1][x2], 2);
			}
		    }
		}

		if (w1[i] == 0 && w2[j] == 0) {
		    errorNone = error;
		} else if (w1[i] == 1 && w2[j] == 0) {
		    errorW1 = error;
		} else if (w1[i] == 0 && w2[j] == 1) {
		    errorW2 = error;
		} else if (w1[i] == 1 && w2[j] == 1) {
		    errorW1W2 = error;
		}

		if (error < minError) {
		    minError = error;
		    opt_w1 = w1[i];
		    opt_w2 = w2[j];
		}

		minErrorMatrix[i][j] = error;

	    }
	}

	double diff = Math.min(Math.min(Math.min(errorW1, errorW2), errorW1W2), errorNone) - minError;

	System.out.printf("%f,%f,%f,%f,%f,%f,%f,%f,%.8f,%f,%f\n", minError, opt_w1, opt_w2, errorNone, errorW1, errorW2, errorW1W2, NBerrorSq, diff,
		CMI, MI);

    } // ends computeExpError

    public static double log2(double num) {
	return (Math.log(num) / Math.log(2));
    }

} // ends class

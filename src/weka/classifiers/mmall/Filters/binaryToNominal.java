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
 * infoStream.java     
 * Code written by: Nayyar Zaidi
 * 
 * Options:
 * -------
 * 
 * Arg1: Path to data file: -T /home/nayyar/Data/datasets_DM/pendigits.arff
 */
package weka.classifiers.mmall.Filters;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public class binaryToNominal {

	private static String m_InputFile;
	private static String m_OutputFile;
	private static String m_Name;
	private static int m_FirstIndex; // Index to be specified from 0
	private static int m_LastIndex;

	public static void main(String[] args) throws Exception {

		setOptions(args);

		File sourceFile = new File(m_InputFile);
		if (!sourceFile.exists()) {
			System.err.println("File " + args[0] + " not found!");
			System.exit(-1);
		} else {
			System.out.println("---- " + m_InputFile + " ----");
		}

		File outputFile = new File(m_OutputFile);

		FileReader fReader = new FileReader(sourceFile);
		BufferedReader reader = new BufferedReader(fReader);
		ArffReader loader = new ArffReader(reader, 100000);

		FileWriter fw = new FileWriter(outputFile);
		BufferedWriter bw = new BufferedWriter(fw);

		Instances structure = loader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);		

		int nAttributes = structure.numAttributes() - 1;
		int nc = structure.numClasses();

		System.out.println("Number of attributes are: " + nAttributes);
		System.out.println("Number of classes are: " + nc);

		/* Modifying and writing header */		
		bw.write("@relation " + m_InputFile + "-BinaryToNominalFilter\n");
		bw.write("\n");

		// Check if the attributes in the range are all binary
		for (int u = m_FirstIndex; u < m_LastIndex; u++) {
			if (structure.attribute(u).numValues() != 2) {
				throw new Exception("Non Binary Attribute in the range. Can not Merge.");
			}
		}

		int numVals = (m_LastIndex - m_FirstIndex) + 1;
		for (int u = 0; u < nAttributes; u++) {

			if (u == m_FirstIndex) {
				bw.write("@attribute " + m_Name + " {");
				for (int i = 0; i < numVals; i++) {
					bw.write(i + "");
					if (i != numVals - 1)
						bw.write(",");

				}
				bw.write("}\n");
			} else if (u > m_FirstIndex && u <= m_LastIndex) {
				// Dont do anything
			} else {
				if (structure.attribute(u).isNumeric()) {
					bw.write("@attribute " + structure.attribute(u).name() + " real\n");
				} else {
					bw.write("@attribute " + structure.attribute(u).name() + " {");
					for (int uval = 0; uval < structure.attribute(u).numValues(); uval++) {
						bw.write(uval + "");

						if (uval != structure.attribute(u).numValues() - 1)
							bw.write(",");

					}
					bw.write("}\n");
				}
			}
		}

		bw.write("@attribute Class {");
		for (int c = 0; c < nc; c++) {
			bw.write(c+"");
			if (c != nc-1)
				bw.write(",");
		}
		bw.write("}\n");

		bw.write("\n@data \n");	

		/* Modifying Data */

		Instance current;
		int N = 0;
		while ((current = loader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();			
			N++;
			
			int val = 0;
			int index = 0;
			for (int i = m_FirstIndex; i < m_LastIndex; i++) {				
				if (current.value(i) == 1)
					val = index;
				index++;
			}

			for (int u = 0; u < nAttributes; u++) {
				
				if (u == m_FirstIndex) {
					bw.write(val + ",");
				} else if (u > m_FirstIndex && u <= m_LastIndex) {
					// Dont do anything
				} else {
					bw.write((int) current.value(u) + ",");
				}
			}
			bw.write(x_C+"");
			bw.write("\n");
		}		

		bw.close();			

		System.out.println("File Writen - " + outputFile);
		System.out.println("Number of instances: " + N);
		System.out.println("All Done");
	}

	public static void setOptions(String[] options) throws Exception {
		String strT = Utils.getOption('I', options);
		if (strT.length() != 0) {
			m_InputFile = strT;
		}
		String strO = Utils.getOption('O', options);
		if (strO.length() != 0) {
			m_OutputFile = strO;
		}
		String strF = Utils.getOption('F', options);
		if (strF.length() != 0) {
			m_FirstIndex = Integer.parseInt(strF);
		}
		String strL = Utils.getOption('L', options);
		if (strL.length() != 0) {
			m_LastIndex = Integer.parseInt(strL);
		}
		String strN = Utils.getOption('N', options);
		if (strN.length() != 0) {
			m_Name = strN;
		}
	}

	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

}

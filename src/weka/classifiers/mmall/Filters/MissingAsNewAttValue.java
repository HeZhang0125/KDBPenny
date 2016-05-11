/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka.classifiers.mmall.Filters;


import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.Sourcable;
import weka.filters.UnsupervisedFilter;

/** 
 <!-- globalinfo-start -->
 * Replaces all missing values for nominal attributes in a dataset with a new value for every attribute that contains missing values.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -unset-class-temporarily
 *  Unsets the class index temporarily before the filter is
 *  applied to the data.
 *  (default: no)</pre>
 * 
 <!-- options-end -->
 * 
 *
 * @author ana (anam.martinez@monash.edu)
 *
 * 
 */
public class MissingAsNewAttValue
  extends Filter
  implements UnsupervisedFilter, Sourcable {

  /** for serialization */
  //static final long serialVersionUID = 8349568310991609867L;
  
  /** vector to indicate if the missing value '?' has been added to a particular attribute */
  private boolean[] m_SeenAtts = null;

  /**
   * Returns a string describing this filter
   *
   * @return a description of the filter suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return "Replaces all missing values for nominal attributes in a dataset with a new value for every attribute that contains missing values..";
  }

  /** 
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  @Override
public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);
    
    // class
    result.enableAllClasses();
    result.enable(Capability.MISSING_CLASS_VALUES);
    result.enable(Capability.NO_CLASS);
    
    return result;
  }

  /**
   * Sets the format of the input instances.
   *
   * @param instanceInfo an Instances object containing the input 
   * instance structure (any instances contained in the object are 
   * ignored - only the structure is required).
   * @return true if the outputFormat may be collected immediately
   * @throws Exception if the input format can't be set 
   * successfully
   */
  @Override
public boolean setInputFormat(Instances instanceInfo) 
       throws Exception {

    super.setInputFormat(instanceInfo);
    setOutputFormat(instanceInfo);
    m_SeenAtts=null;
    //m_SeenAtts=new boolean[instanceInfo.numAttributes()];
    for (int i = 0; i < instanceInfo.numAttributes(); i++) {
        if (!getInputFormat().attribute(i).isNominal()) {
            throw new UnsupportedAttributeTypeException("Not nominal attribute found.");
        }
    }
    return false;
  }

  /**
   * Input an instance for filtering. Filter requires all
   * training instances be read before producing output.
   *
   * @param instance the input instance
   * @return true if the filtered instance may now be
   * collected with output().
   * @throws IllegalStateException if no input format has been set.
   */
  @Override
public boolean input(Instance instance) {

    Instance 	newInstance = new DenseInstance(instance);
    newInstance.setDataset(instance.dataset());
    
    
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      m_NewBatch = false;
    }
    
    if (m_SeenAtts != null) {
      convertInstance(instance);
      return true;
    }

    bufferInput(instance);
    return false;
  }
  
  
  /**
   * Convert a single instance over. The converted instance is added to
   * the end of the output queue.
   *
   * @param instance the instance to convert
   */
  protected void convertInstance(Instance instance) {
      
    String extraValue = "?";
    Instance 	newInstance = new DenseInstance(instance);
    
    for (int i = 0; i < getInputFormat().numAttributes(); i++) {
        if (instance instanceof SparseInstance) {
            if(instance.isMissingSparse(i)){     
                throw new IllegalStateException("Filter not yet implemented for sparse format");
           }
        }
        else{
            if(instance.isMissing(i)){
                newInstance.setValue(i, getOutputFormat().attribute(i).indexOfValue(extraValue));
            }
        }
    }  

    newInstance.setDataset(getOutputFormat());
    copyValues(newInstance, false, instance.dataset(), getOutputFormat());
    newInstance.setDataset(getOutputFormat());
    push(newInstance);
  }

  
    @Override
	public boolean batchFinished() {

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_SeenAtts == null) {

      setOutputFormat();

      // If we implement saving cutfiles, save the cuts here

      // Convert pending input instances
      for(int i = 0; i < getInputFormat().numInstances(); i++) {
	convertInstance(getInputFormat().instance(i));
      }
    }
    flushInput();

    m_NewBatch = true;
    return (numPendingOutput() != 0);
  }
    
    /**
   * Set the output format. 
   */
  protected void setOutputFormat() {
    Instances	instNew=new Instances(getInputFormat());
    String extraValue = "?";
    Attribute   attNew;
    FastVector	atts = new FastVector();
    
    m_SeenAtts = m_SeenAtts=new boolean[instNew.numAttributes()];
    for (int i = 0; i < instNew.numAttributes(); i++) {
        m_SeenAtts[i]=false;
    }
    

       for (int i = 0; i < getInputFormat().numAttributes(); i++) {
           Attribute att = getInputFormat().attribute(i);
           for (int e = 0; e < instNew.numInstances(); e++) {
                Instance instance=instNew.get(e);
                if (instance instanceof SparseInstance) {
                    if(instance.isMissingSparse(i)){     
                        throw new IllegalStateException("Filter not yet implemented for sparse format");
                   }
                }
                else{
                    if(instance.isMissing(i)){
                        if(!m_SeenAtts[i]){
                            FastVector values = new FastVector();
                            for (int j = 0; j < att.numValues(); j++){
                              values.addElement(att.value(j));
                            }
                            values.addElement(extraValue);
                            attNew = new Attribute(att.name(), values);

                            atts.addElement(attNew);

                            m_SeenAtts[i]=true;
                        }                                                           
                    }
                }
           }
           if(!m_SeenAtts[i]){
               atts.addElement(att);
           }
    }

    instNew = new Instances(getInputFormat().relationName(), atts, 0);
    instNew.setClassIndex(getInputFormat().classIndex());
    setOutputFormat(instNew);
  
  }
  
  
  /**
   * Returns a string that describes the filter as source. The
   * filter will be contained in a class with the given name (there may
   * be auxiliary classes),
   * and will contain two methods with these signatures:
   * <pre><code>
   * // converts one row
   * public static Object[] filter(Object[] i);
   * // converts a full dataset (first dimension is row index)
   * public static Object[][] filter(Object[][] i);
   * </code></pre>
   * where the array <code>i</code> contains elements that are either
   * Double, String, with missing values represented as null. The generated
   * code is public domain and comes with no warranty.
   *
   * @param className   the name that should be given to the source class.
   * @param data	the dataset used for initializing the filter
   * @return            the object source described by a string
   * @throws Exception  if the source can't be computed
   */
  @Override
public String toSource(String className, Instances data) throws Exception {
    StringBuffer        result;
    boolean[]		numeric;
    boolean[]		nominal;
    String[]		modes;
    double[]		means;
    int			i;
    
    result = new StringBuffer();
    
    // determine what attributes were processed
    numeric = new boolean[data.numAttributes()];
    nominal = new boolean[data.numAttributes()];
    modes   = new String[data.numAttributes()];
    means   = new double[data.numAttributes()];
    for (i = 0; i < data.numAttributes(); i++) {
      numeric[i] = (data.attribute(i).isNumeric() && (i != data.classIndex()));
      nominal[i] = (data.attribute(i).isNominal() && (i != data.classIndex()));
     
    }
    
    result.append("class " + className + " {\n");
    result.append("\n");
    result.append("  /** lists which numeric attributes will be processed */\n");
    result.append("  protected final static boolean[] NUMERIC = new boolean[]{" + Utils.arrayToString(numeric) + "};\n");
    result.append("\n");
    result.append("  /** lists which nominal attributes will be processed */\n");
    result.append("  protected final static boolean[] NOMINAL = new boolean[]{" + Utils.arrayToString(nominal) + "};\n");
    result.append("\n");
    result.append("  /** the means */\n");
    result.append("  protected final static double[] MEANS = new double[]{" + Utils.arrayToString(means).replaceAll("NaN", "Double.NaN") + "};\n");
    result.append("\n");
    result.append("  /** the modes */\n");
    result.append("  protected final static String[] MODES = new String[]{");
    for (i = 0; i < modes.length; i++) {
      if (i > 0)
	result.append(",");
      if (nominal[i])
	result.append("\"" + Utils.quote(modes[i]) + "\"");
      else
	result.append(modes[i]);
    }
    result.append("};\n");
    result.append("\n");
    result.append("  /**\n");
    result.append("   * filters a single row\n");
    result.append("   * \n");
    result.append("   * @param i the row to process\n");
    result.append("   * @return the processed row\n");
    result.append("   */\n");
    result.append("  public static Object[] filter(Object[] i) {\n");
    result.append("    Object[] result;\n");
    result.append("\n");
    result.append("    result = new Object[i.length];\n");
    result.append("    for (int n = 0; n < i.length; n++) {\n");
    result.append("      if (i[n] == null) {\n");
    result.append("        if (NUMERIC[n])\n");
    result.append("          result[n] = MEANS[n];\n");
    result.append("        else if (NOMINAL[n])\n");
    result.append("          result[n] = MODES[n];\n");
    result.append("        else\n");
    result.append("          result[n] = i[n];\n");
    result.append("      }\n");
    result.append("      else {\n");
    result.append("        result[n] = i[n];\n");
    result.append("      }\n");
    result.append("    }\n");
    result.append("\n");
    result.append("    return result;\n");
    result.append("  }\n");
    result.append("\n");
    result.append("  /**\n");
    result.append("   * filters multiple rows\n");
    result.append("   * \n");
    result.append("   * @param i the rows to process\n");
    result.append("   * @return the processed rows\n");
    result.append("   */\n");
    result.append("  public static Object[][] filter(Object[][] i) {\n");
    result.append("    Object[][] result;\n");
    result.append("\n");
    result.append("    result = new Object[i.length][];\n");
    result.append("    for (int n = 0; n < i.length; n++) {\n");
    result.append("      result[n] = filter(i[n]);\n");
    result.append("    }\n");
    result.append("\n");
    result.append("    return result;\n");
    result.append("  }\n");
    result.append("}\n");
    
    return result.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  @Override
public String getRevision() {
    return RevisionUtils.extract("$Revision: 8034 $");
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain arguments to the filter: 
   * use -h for help
   */
  public static void main(String [] argv) {
    runFilter(new MissingAsNewAttValue(), argv);
  }
}


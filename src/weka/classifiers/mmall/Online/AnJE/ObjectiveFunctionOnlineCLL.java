package weka.classifiers.mmall.Online.AnJE;

import weka.classifiers.mmall.Online.AnJE.wdAnJEOnline;
import weka.core.Instance;
import weka.core.Instances;

public abstract class ObjectiveFunctionOnlineCLL {
	
	protected final wdAnJEOnline algorithm;
	
	public ObjectiveFunctionOnlineCLL(wdAnJEOnline algorithm) {
		this.algorithm = algorithm;
	}

	abstract public void update(Instance instance, int t, double[] results);

	abstract public void update(Instances m_Instances);
	
	public void finish(){
		
	}
	
}

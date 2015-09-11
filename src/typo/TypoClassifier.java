package typo;

import java.util.*;
import utils.Utils;

public class TypoClassifier {
	public String[] stateStr;
	HashMap<String, Integer> stateDict;
	int stateNum;
	int targetNum;
	double[] feature;
	double[][] weights;
	
	public int stateIndex(String str) {
		return stateDict.containsKey(str) ? stateDict.get(str) : -1;
	}
	
	public int featIndex(int i, int j) {
		return i * stateNum + j;
	}
	
	public void initialize() {
		stateDict = new HashMap<String, Integer>();
		stateDict.put("s", 0);
		stateDict.put("e", 1);
		stateDict.put("0", 2);
		stateDict.put("1", 3);
		stateNum = stateDict.size();
		
		stateStr = new String[stateNum];
		for (String s : stateDict.keySet()) {
			stateStr[stateDict.get(s)] = s;
		}
		
		feature = new double[stateNum * stateNum];
		targetNum = 2;
		weights = new double[targetNum][stateNum * stateNum];
		
		// For toy example.
		weights[1][featIndex(stateIndex("0"), stateIndex("e"))] = 0.5;
		weights[0][featIndex(stateIndex("1"), stateIndex("e"))] = 5;
		weights[0][featIndex(stateIndex("0"), stateIndex("0"))] = 1;
		weights[1][featIndex(stateIndex("1"), stateIndex("1"))] = 0.5;
		weights[1][featIndex(stateIndex("1"), stateIndex("0"))] = 1;
	}
	
	public void constructFeature(double[][] data) {
		for (int i = 0; i < stateNum; ++i) {
			for (int j = 0; j < stateNum; ++j) {
				feature[featIndex(i, j)] = data[i][j];
			}
		}
	}
	
	public void softmax(double[] score) {
		double max = Utils.max(score);
		if (max == Double.NEGATIVE_INFINITY) {
			for (int i = 0; i < score.length; ++i)
				score[i] = 0.0;
		}
		else {
			for (int i = 0; i < score.length; ++i)
				score[i] -= max;
			double sum = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < score.length; ++i)
				sum = Utils.logSumExp(sum, score[i]);
			for (int i = 0; i < score.length; ++i)
				score[i] = Math.exp(score[i] - sum);
		}
	}
	
	public double[] computeScore(double[][] data) {
		constructFeature(data);
		double[] score = new double[targetNum];
		for (int i = 0; i < targetNum; ++i)
			score[i] = Utils.dotsum(weights[i], feature);
		softmax(score);
		return score;
	}
	
	public double entropy(double[] prob) {
		double e = 0.0;
		for (int i = 0; i < prob.length; ++i)
			e += prob[i] == 0.0 ? 0.0 : prob[i] * Math.log(prob[i]);
		return e;
	}
	
	public double[][] gradient(double[][] data) {
		double[] score = computeScore(data);
		
		// A_i = 1 + log(p_i)
		double[] entropyGradient = new double[targetNum];
		for (int i = 0; i < targetNum; ++i)
			entropyGradient[i] = score[i] == 0.0 ? Double.NEGATIVE_INFINITY : (1 + Math.log(score[i]));
		
		// B_{ij} = p_i(1-p_i) or -p_ip_j
		double[][] softmaxGradient = new double[targetNum][targetNum];
		for (int i = 0; i < targetNum; ++i)
			for (int j = 0; j < targetNum; ++j) {
				if (i == j)
					softmaxGradient[i][j] = score[i] * (1 - score[i]);
				else
					softmaxGradient[i][j] = -score[i] * score[j];
			}
		
		// C = A * B
		double[] backGradient = new double[targetNum];
		for (int i = 0; i < targetNum; ++i) {
			for (int j = 0; j < targetNum; ++j)
				backGradient[i] += entropyGradient[j] * softmaxGradient[j][i];
		}
		
		// D = C * weights
		double[] featureGradient = new double[stateNum * stateNum];
		for (int i = 0; i < featureGradient.length; ++i) {
			for (int j = 0; j < targetNum; ++j)
				featureGradient[i] += backGradient[j] * weights[j][i];
		}
		
		// Feature gradient to data gradient
		double[][] dataGradient = new double[stateNum][stateNum];
		for (int i = 0; i < stateNum; ++i)
			for (int j = 0; j < stateNum; ++j) {
				dataGradient[i][j] = featureGradient[featIndex(i, j)];
			}
		
		return dataGradient;
	}
	
	public void train() {
		double[][] data = new double[stateNum][stateNum];
		Random r = new Random();
		for (int i = 0; i < stateNum; ++i) {
			for (int j = 0; j < stateNum; ++j) {
				data[i][j] = r.nextDouble();
			}
		}
		
		double learnRate = 0.5;
		for (int i = 0; i < 100; ++i) {
			System.out.println(entropy(computeScore(data)));
			double[][] g = gradient(data);
			for (int j = 0; j < stateNum; ++j) {
				for (int k = 0; k < stateNum; ++k) {
					data[j][k] += learnRate * g[j][k];
				}
			}
			if (i % 10 == 0)
				learnRate *= 0.9;
		}
	}
	
	public static void main(String[] args) {
		TypoClassifier typo = new TypoClassifier();
		typo.initialize();
		/*
		String[] s = {"a", "c", "a", "c", "b"};
		for (int a = 0; a <= 1; ++a)
			for (int b = 0; b <= 1; ++b)
				for (int c = 0; c <= 1; ++c) {
					System.out.println("a: " + a + "; b:" + b + " c: " + c);
					String[] code = new String[s.length + 2];
					code[0] = "s"; code[code.length - 1] = "e";
					for (int i = 0; i < s.length; ++i) {
						if (s[i].equals("a"))
							code[i + 1] = new Integer(a).toString();
						else if (s[i].equals("b"))
							code[i + 1] = new Integer(b).toString();
						else if (s[i].equals("c"))
							code[i + 1] = new Integer(c).toString();
					}
					double[][] data = new double[typo.stateNum][typo.stateNum];
					for (int i = 0; i < code.length - 1; ++i) {
						data[typo.stateIndex(code[i])][typo.stateIndex(code[i + 1])] += 1.0;
					}
					double[] score = typo.computeScore(data);
					System.out.println(score[0] + " " + score[1] + " " + typo.entropy(score));
					double[][] grad = typo.gradient(data);
					for (int i = 0; i < typo.stateNum; ++i) {
						for (int j = 0; j < typo.stateNum; ++j)
							System.out.print(String.format("%.3f/%.3f\t", grad[i][j], data[i][j]));
						System.out.println();
					}
				}
				*/
		typo.train();
	}
}

package misc;

import java.util.*;
import java.io.*;
import utils.Utils;

public class Processor {
	
	public static HashSet<String> targetLang = new HashSet<String>();

	public static HashMap<String, String> readMap(String fileName) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		HashMap<String, String> map = new HashMap<String, String>();
		String str = null;
		while ((str = br.readLine()) != null) {
			String[] data = str.split("\\s+");
			Utils.Assert(!data[0].isEmpty() && !data[1].isEmpty());
			map.put(data[0], data[1]);
		}
		br.close();
		return map;
	}
	
	public static void generateCONLLFile(String inputName, String outputName, HashMap<String, String> map) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(inputName));
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputName));
		String str = null;
		while ((str = br.readLine()) != null) {
			String line = "";
			while (!str.isEmpty()) {
				String[] data = str.split("\\s+");
				if ((data[4].contains("Head") || data[4].contains("property") || data[4].contains("epistemics")) && data[1].contains(":")) {
					data[4] = data[1].substring(0, data[1].indexOf(":"));
				}
				else if (data[3].equals("MWU")) {
					data[4] = data[4].substring(data[4].lastIndexOf("_") + 1);
				}
				else if (data[3].equals("--") && data[4].equals("--")) {
					System.out.print("skip ");
					str = br.readLine();
					continue;
				}
				else if ((data[3].equals("ec") && data[4].equals("ec"))
						|| (data[3].equals("?") && data[4].equals("?"))) {
					System.out.print("skip ");
					str = br.readLine();
					continue;
				}
				else {
					if (!map.containsKey(data[4])) {
						System.out.println(str);
					}
					Utils.Assert(map.containsKey(data[4]));
				}
				line += map.get(data[4]) + " ";
				str = br.readLine();
			}
			bw.write(line.trim() + "\n");
		}
		br.close();
		bw.close();
	}
	
	public static void mapCONLLTag() throws IOException {
		String[] langs = {"arabic07", "arabic", "basque07", "bulgarian", "catalan07", 
				"chinese07", "chinese", "czech07", "czech", "danish", "dutch", "english07", "german", "greek07",
				"hungarian07", "italian07", "japanese", "portuguese", "slovene", "spanish", "swedish", "turkish07", "turkish"};
		for (int i = 0; i < langs.length; ++i) {
			System.out.println(langs[i]);
			HashMap<String, String> map = readMap("../data/CONLL_06_07/" + langs[i] + ".uni.map");
			String output = langs[i].contains("07") ? langs[i] : (langs[i] + "06"); 
			generateCONLLFile("../data/CONLL_06_07/" + langs[i] + ".train", "data/" + output + ".uni.train", map);
			generateCONLLFile("../data/CONLL_06_07/" + langs[i] + ".test", "data/" + output + ".uni.test", map);
		}
	}
	
	public static void generateUDFile(String inputName, String outputName, HashMap<String, String> map) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(inputName));
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputName));
		String str = null;
		while ((str = br.readLine()) != null) {
			if (str.startsWith("#"))
				continue;
			String line = "";
			while (!str.isEmpty()) {
				String[] data = str.split("\\s+");
				if (data[4].equals("IN")) {
					data[3] = "ADP";
				}
				else if (data[3].equals("_")) {
					str = br.readLine();
					continue;
				}
				else {
					if (!map.containsKey(data[3])) {
						System.out.println(str);
					}
					Utils.Assert(map.containsKey(data[3]));
				}
				line += map.get(data[3]) + " ";
				str = br.readLine();
			}
			bw.write(line.trim() + "\n");
		}
		br.close();
		bw.close();
	}
	
	public static void mapUDTag() throws IOException {
		String[] langs = {"Basque", "Bulgarian", "Croatian", "Czech", "Danish", 
				"English", "Finnish", "Finnish-FTB", "French", "German", "Greek",
				"Hebrew", "Hungarian", "Indonesian", "Irish", "Italian", "Persian", "Spanish", "Swedish"};
		HashMap<String, String> map = readMap("data/tmp/ud-treebanks-v1.1/map.txt");
		for (int i = 0; i < langs.length; ++i) {
			System.out.println(langs[i]);
			File dir = new File("data/tmp/ud-treebanks-v1.1/UD_" + langs[i]);
			for (File f : dir.listFiles()) {
				String s = f.getAbsolutePath();
				if (s.contains("train.conllu")) {
					generateUDFile(s, "data/" + langs[i] + "_ud.uni.train", map);
				}
				else if (s.contains("test.conllu")) {
					generateUDFile(s, "data/" + langs[i] + "_ud.uni.test", map);
				}
			}
		}
	}
	
	public static void generateUT2File(String inputName, String outputName) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(inputName));
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputName));
		String str = null;
		while ((str = br.readLine()) != null) {
			String line = "";
			while (!str.isEmpty()) {
				String[] data = str.split("\\s+");
				if (!data[3].equals("_")) {
					line += data[3] + " ";
				}
				str = br.readLine();
			}
			bw.write(line.trim() + "\n");
		}
		br.close();
		bw.close();
	}
	
	public static void mapUT2Tag() throws IOException {
		String[] langs = {"de", "en", "es", "fr", "id", 
				"it", "ja", "ko", "pt-br", "sv"};
		for (int i = 0; i < langs.length; ++i) {
			System.out.println(langs[i]);
			generateUT2File("../TensorTransfer/data/universal_treebanks_v2.0/std/" + langs[i] +"/" + langs[i] + "-universal-train.conll", 
					"data/" + langs[i] + "_ut2.uni.train");
			generateUT2File("../TensorTransfer/data/universal_treebanks_v2.0/std/" + langs[i] +"/" + langs[i] + "-universal-test.conll", 
					"data/" + langs[i] + "_ut2.uni.test");
		}
	}
	
	public static HashMap<String, Integer> getLanguageFeature(int featID) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader("data/feature.txt"));
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		String str = br.readLine();
		while ((str = br.readLine()) != null) {
			String[] data = str.split("\\s+");
			map.put(data[0], Integer.parseInt(data[featID + 1]));
		}
		br.close();
		return map;
	}
	
	public static String[] randomShuffle(ArrayList<String> list) {
		String[] data = new String[list.size()];
		boolean[] used = new boolean[list.size()];
		Random r = new Random(0);
		for (int i = 0; i < data.length; ++i) {
			int idx = r.nextInt(data.length);
			while (used[idx]) {
				idx = (idx + 1) % data.length;
			}
			data[i] = list.get(idx);
			used[idx] = true;
		}
		return data;
	}
	
	public static String[] getDataOfFeatValue(HashMap<String, Integer> lang, int featValue, String type) throws IOException {
		ArrayList<String> list = new ArrayList<String>();
		for (String l : lang.keySet()) {
			if (!lang.get(l).equals(featValue))
				continue;
			if (targetLang.contains(l))
				continue;
			System.out.println(l + ":" + featValue); 
			BufferedReader br = new BufferedReader(new FileReader("data/" + l + ".uni." + type));
			String str = null;
			while ((str = br.readLine()) != null) {
				list.add(str);
			}
			br.close();
		}
		
		return randomShuffle(list);
	}
	
	public static String[] getDataOfLang(String lang, String type) throws IOException {
		ArrayList<String> list = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader("data/" + lang + ".uni." + type));
		String str = null;
		while ((str = br.readLine()) != null) {
			list.add(str);
		}
		br.close();
		
		return randomShuffle(list);
	}
	
	public static String[][][] getData(HashMap<String, Integer> lang, int featValue) throws IOException {
		// [train/test][featValue/lang][data]
		String[][][] data = new String[2][][];
		data[0] = new String[featValue][];
		for (int i = 0; i < featValue; ++i) {
			data[0][i] = getDataOfFeatValue(lang, i, "train");
		}
		data[1] = new String[lang.size()][];
		//data[1] = new String[targetLang.size()][];
		int i = 0;
		for (String l : lang.keySet()) {
//		for (String l : targetLang) {
			data[1][i] = getDataOfLang(l, "test");
			i++;
		}
		return data;
	}
	
	public static void addCnt(HashMap<Integer, Integer> map, int featID) {
		if (map.containsKey(featID)) {
			map.put(featID, map.get(featID) + 1);
		}
		else {
			map.put(featID, 1);
		}
	}
	
	public static String getCoarse(String t) {
		if (t.equals("NOUN") || t.equals("PRON"))
			return "CAT1";
		else if (t.equals("START") || t.equals("END") || t.equals("CONJ") || t.equals("PUNC"))
			return "CAT2";
		else
			return "NULL";
	}
	
	public static void addBigram(String t1, String t2, HashMap<String, Integer> featIndex, HashMap<Integer, Integer> featCnt) {
			// bigram
			String featStr = "";
			int featID = 0;
			featStr = t1 + "," + t2;
			if (!featIndex.containsKey(featStr))
				featIndex.put(featStr, featIndex.size());
			featID = featIndex.get(featStr);
			addCnt(featCnt, featID);
			
		String key = "ADP";
		if (t1.equals(key) || t2.equals(key)) {
			// coarse
			if (t1.equals(key)) {
				String t2c = getCoarse(t2);
				if (!t2c.equals("NULL")) {
					featStr = t1 + "," + t2c;
					if (!featIndex.containsKey(featStr))
						featIndex.put(featStr, featIndex.size());
					featID = featIndex.get(featStr);
					addCnt(featCnt, featID);
				}
			}
			if (t2.equals(key)) {
				String t1c = getCoarse(t1);
				if (!t1c.equals("NULL")) {
					featStr = t1c + "," + t2;
					if (!featIndex.containsKey(featStr))
						featIndex.put(featStr, featIndex.size());
					featID = featIndex.get(featStr);
					addCnt(featCnt, featID);
				}
			}
		}
		
	}
	
	public static double[][] generateFeature(int numExample, String[] data, 
			HashMap<String, Integer> featIndex) {
		System.out.println("size: " + data.length);
		String[][] tags = new String[data.length][];
		int numToken = 0;
		for (int i = 0 ; i < data.length; ++i) {
			tags[i] = data[i].split(" ");
			numToken += tags[i].length;
		}
		double[][] feature = new double[numExample][];
		int tokenPerExample = numToken / numExample;
		int idx = 0;
		for (int i = 0; i < numExample; ++i) {
			int currToken = 0;
			HashMap<Integer, Integer> featCnt = new HashMap<Integer, Integer>();
			while (currToken < tokenPerExample && idx < tags.length) {
				String[] sentence = tags[idx];
				addBigram("START", sentence[0], featIndex, featCnt);
				for (int j = 0; j < sentence.length - 1; ++j)
					addBigram(sentence[j], sentence[j + 1], featIndex, featCnt);
				addBigram(sentence[sentence.length - 1], "END", featIndex, featCnt);
				currToken += tags[idx].length;
				idx++;
			}
			feature[i] = new double[featIndex.size()];
			for (Integer featID : featCnt.keySet()) {
				feature[i][featID] = featCnt.get(featID).doubleValue() / currToken * 10.0;
			}
		}
		return feature;
	}
	
	public static void outputFeature(BufferedWriter bw, double[][] feature, int category) throws IOException {
		for (int i = 0; i < feature.length; ++i) {
			bw.write("" + (category + 1));
			for (int j = 0; j < feature[i].length; ++j) {
				if (feature[i][j] > 1e-6)
					bw.write(String.format(" %d:%.5f", j + 1, feature[i][j]));
			}
			bw.newLine();
		}
	}
	
	public static void generateFeature() throws IOException {
		targetLang.add("italian07");
		targetLang.add("Italian_ud");
		targetLang.add("it_ut2");
		
		int featID = 3;
		int numValue = 3;
		int trainExample = 150;
		int testExample = 10;
		HashMap<String, Integer> languageFeature = getLanguageFeature(featID);
		String[][][] data = getData(languageFeature, numValue);
		BufferedWriter bw_train = new BufferedWriter(new FileWriter("data/svm.train"));
		HashMap<String, Integer> featIndex = new HashMap<String, Integer>();
		for (int i = 0; i < numValue; ++i) {
			double[][] feature = generateFeature(trainExample, data[0][i], featIndex);
			outputFeature(bw_train, feature, i);
		}
		BufferedWriter bw_test = new BufferedWriter(new FileWriter("data/svm.test"));
		int i = 0;
		for (String l : languageFeature.keySet()) {
//		for (String l : targetLang) {
			double[][] feature = generateFeature(1, data[1][i], featIndex);
			outputFeature(bw_test, feature, languageFeature.get(l));
			i++;
		}
		bw_train.close();
		bw_test.close();
		
		for (String s : featIndex.keySet()) {
			System.out.println(s + "\t" + featIndex.get(s));
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			//mapCONLLTag();
			//mapUDTag();
			//mapUT2Tag();
			generateFeature();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}

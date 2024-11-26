package main;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DataSetCreator {

	public static final int RANGE = 80; // FeatureVector count per concept: (RANGE * 0.2 + 1)^2

	private final Random random;

	private final List<FeatureVector> trainingsSet;

	private final List<FeatureVector> testSet;

	DataSetCreator(int trainingsSetSizePerConcept, long seed) {
		this.random = new Random(seed);

		this.trainingsSet = new ArrayList<>();
		this.testSet = new ArrayList<>();

		for (Concept concept : Concept.values()) {

			List<String> usedNames = new ArrayList<>();

			for (int i = 0; i < trainingsSetSizePerConcept; i++) {
				String xy;

				do {
					int brightness = this.random.nextInt(5) - 2;
					String bStr = Util.getBrightness(brightness);
					int randNum1 = nextRandom();
					int randNum2 = nextRandom();
					xy = bStr + "X" + randNum1 + "Y" + randNum2;
				} while (usedNames.contains(xy));

				usedNames.add(xy);

				FeatureVector fv = FileManager.readFile(concept.name() + xy);
				trainingsSet.add(fv);
			}

			for (int b = -2; b <= 2; b++) {
				String bStr = Util.getBrightness(b);

				for (int i = -RANGE; i <= RANGE; i += 10) {
					for (int k = -RANGE; k <= RANGE; k += 10) {

						String xy = bStr + "X" + i + "Y" + k;

						if (!usedNames.contains(xy)) {
							FeatureVector fv = FileManager.readFile(concept.name() + xy);
							testSet.add(fv);
						}
					}
				}
			}
		}

		Collections.shuffle(trainingsSet);
		Collections.shuffle(testSet);
	}

	private int nextRandom() {
		int randNum = this.random.nextInt(RANGE / 5 + 1);
		randNum -= (RANGE / 10);
		randNum *= 10;
		return randNum;
	}

	List<FeatureVector> getTrainingsData() {
		return this.trainingsSet;
	}

	List<FeatureVector> getTestData() {
		return this.testSet;
	}

}
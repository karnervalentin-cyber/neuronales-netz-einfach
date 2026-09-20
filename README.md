# Neuronales Netz für resultierende Kräfte

🚧 **Work in Progress** — das Netz lernt aktuell noch nicht. Ich baue hier
öffentlich und dokumentiere den Weg mit, inklusive der Sackgassen.

## Worum es geht

Ein neuronales Netz, vollständig von Hand in reinem Python gebaut — ohne NumPy,
ohne PyTorch, ohne TensorFlow. Einzige Imports: `random` und `math`.

Die Aufgabe: Aus den drei Komponenten einer Kraft (Fx, Fy, Fz) den Betrag der
resultierenden Kraft vorhersagen. Also `F = √(Fx² + Fy² + Fz²)` allein aus 30
Trainingsdatensätzen lernen, ohne die Formel je einzuprogrammieren.

## Warum von Grund auf?

Mit PyTorch wäre das in zehn Zeilen erledigt — und ich hätte nichts darüber
gelernt, was dabei eigentlich passiert. Hier ist jeder Baustein selbst gebaut:
Neuron, Layer, Forward Pass, Loss-Funktion, Gradient Descent.

## Aufbau

- **Architektur:** 3 → 18 → 8 → 1
- **Aktivierung:** ReLU
- **Loss:** Summe der quadratischen Fehler
- **Optimierung:** Gradient Descent, Lernrate 0.1

Die Gradienten berechne ich **numerisch** über den Differenzenquotienten
(`(f(x+h) - f(x)) / h`, h = 10⁻⁶) statt über Backpropagation. Für jedes einzelne
Gewicht wird der gesamte Loss zweimal neu bestimmt — langsam, aber es macht ohne
Umwege sichtbar, was ein Gradient ist: die Frage, wie sich der Fehler ändert,
wenn man an genau einer Schraube minimal dreht.

## Aktueller Stand

Der Loss bleibt über alle Epochen konstant bei 7945.43 und die Ausgabe bei 0 —
egal welche Eingabe. Das Netz lernt also nichts. Zwei Ursachen, die ich bisher
gefunden habe:

**1. Die Inputs kommen nie im Netz an.** `netz_berechnung(inputs)` nimmt zwar ein
Argument entgegen, benutzt es aber nirgends. `layer_a.input_layer` zeigt
dauerhaft auf die Liste `[0, 0, 0]` vom Programmstart. Das Netz sieht bei jedem
Datensatz die gleichen drei Nullen — und lässt sich damit nicht einmal ein
Mittelwert lernen.

**2. Totes Ausgabeneuron.** Auch die letzte Schicht läuft durch ReLU. Sobald die
gewichtete Summe negativ ist, ist die Ausgabe 0 und die Ableitung ebenfalls 0.
Der Gradient Descent hat dann keine Information mehr, in welche Richtung er
gehen soll — das Neuron ist tot und bleibt es.

## Nächste Schritte

- [ ] Inputs korrekt in die erste Schicht durchreichen
- [ ] Ausgabeschicht linear statt ReLU
- [ ] Lernrate neu einstellen (0.1 dürfte mit echten Eingabewerten zu groß sein)
- [ ] Backpropagation implementieren und gegen den numerischen Ansatz benchmarken
- [ ] Vektorisierung mit NumPy

## Verwendung

```bash
python Neuronales_Netz_Res_Kraft_neu.py
```

Trainiert 20 Epochen und fragt danach interaktiv nach Fx, Fy und Fz.

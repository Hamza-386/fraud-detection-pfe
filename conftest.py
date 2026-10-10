import os
import sys

# Garantit que la racine du projet est dans sys.path, afin que
# `from api.main import app` fonctionne quel que soit le répertoire
# depuis lequel pytest est lancé.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

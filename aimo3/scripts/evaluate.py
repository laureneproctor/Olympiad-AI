# Input: model checkpoint + evaluation split
# - Does:
#       - pass@1 with greedy generation
#       - maj@N with sampling + majority vote
#       - reports format validity and agreement stats
# Output:
# - printed metrics + optional runs/report.json
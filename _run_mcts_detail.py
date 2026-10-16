import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    import profiler
    sys.argv = ['profiler', '--mcts-detail', '--gpu', '--workers', '12', '--runs', '1']
    profiler.main()

if __name__ == '__main__':
    main()